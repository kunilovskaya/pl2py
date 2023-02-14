#!/usr/bin/perl
# this is the original script from /data/resources/corpora/bin/ @134.96.90.14
# I added testprints of intermediary variables.
# TESTRUN:
# perl genzelcharniak-vrt_v0.2.1.pl input/brown_exerpt_A-1.vrt output/perl_BROWN-SPR.vrt

use 5.010;

use strict;

use POSIX;
use Getopt::Long qw(GetOptions);
use Encode qw(decode encode);

# parameters
my $lambda = 1.0;  # mixing parameter for ngram-prob vs. cache-prob
my $gamma  = 0.9;  # mixing parameter for Jelinek-Mercer smoothing of cache-prob
my $tau    = 1.0;  # decay parameter for cache-prob (only 1 or 0.99 usually makes sense)
my $maxngram = 3;  # context length for ngrams (with a large corpus such as rsc, don't choose 3 for memory reasons)
my $cross_name = "srp"; # name for cross entropy value in sentence tag
my $ent_name = "srplocal";     # name for entropy value in sentence tag
my $opt_nocross = 0;
my $opt_noent = 0;

# Genzel/Charniak use: $lambda = 1.0 and 0.9, $gamma = 1.0, $tau = 1.0, $maxngram = 3.

GetOptions(
	"lambda=f" => \$lambda,
	"gamma=f" => \$gamma,
	"tau=f" => \$tau,
	"maxngram=i" => \$maxngram,
	"crossname|cn=s" => \$cross_name,
	"entname|en=s" => \$ent_name,
	"nocross" => \$opt_nocross,
	"noent" => \$opt_noent,
	"nocross" => \$opt_nocross,
	) or die "Incorrect usage!\n";

die "Options --nocross and --noent are mutually exclusive!" if $opt_nocross and $opt_noent;

# Annotate a vrt file with bits per word (surprise) according to
# (1) The cross entropy H(Doc|Rest)
# (2) The entropy H(Doc)
# where Doc is the language model for the document at hand, Rest for the rest of the corpus
# Requires sentence tags <s>, </s> to calculate the entropy per sentence

die "Usage: perl genzelcharniak-vrt.pl <vrt-infile> <vrt-outfile>\n".
	"\t[--lambda=f] mixing parameter for ngram-prob vs. cache-prob; default=1.0\n".
	"\t[--gamma=f] mixing parameter for Jelinek-Mercer smoothing of cache-prob; default=0.9\n".
	"\t[--tau=f] decay parameter for cache-prob (only 1 or 0.99 usually makes sense); default=1.0\n".
	"\t[--maxngram=i] context length for ngrams (with a large corpus such as rsc, don't choose 3 for memory reasons); default=3\n".
	"\t[--crossname|cn=s] name for cross entropy value in sentence tag; default=cross\n".
	"\t[--entname|en=s] name for entropy value in sentence tag\n".
	"\t[--noent] do not annotate entropy rate\n"
	unless @ARGV == 2;

my $infile  = $ARGV[0];
my $outfile = $ARGV[1];

# global variables
my %ngrams;       # hash for corpus ngram count, indexed by word\rword\rword
my %ngramtypes;   # hash for corpus ngram type count (for Witten-Bell smoothing)
my %docngrams;    # hash for document ngram count
my %docngramtypes;     # hash for document ngram type count
my %restngramtypes;    # hash for ngram type count in corpus - current document
my %terms;             # hash for corpus term count
my %docterms;          # hash for document term count
my %cacheterms;        # hash for cache term count

my $tokens;            # corpus token count
my $cachetokens;       # cache token count
my $types;             # corpus type count

my @prevterms = ();    # buffer for ngram context

my @lines = ();

my $sumlambda   = 0;
my $countlambda = 0;

my %lastocc;   # hash for last occurrence of a term in document (used for decay of cache)
my $cachecount = 0;

# first pass: compute term frequencies, to transform terms with only 1 occurrence to unknown.

collectterms($infile);
my $size_uni = keys %terms;
print "Types (terms-hash): $size_uni\n";
# second pass: compute ngram frequencies
collectngrams($infile);

{my $size_all_ngrams = keys %ngrams;
print "Collected ngrams (all orders + _, descending freq): $size_all_ngrams\n"};

{my $size = keys %ngramtypes;
print "Collected uni- and bigram contexts aka ngramtypes: $size\n"};

# third pass: compute cache frequencies and bits, etc., two passes per document
# two passes per document: First compute local ngram frequencies, then compute and output bits.

# Open a file for writing
# open(my $fh, '>', 'cr_truest_ngramtypes.txt') or die "Could not open file: $!";
#
# # Loop through the hash and print key-value pairs to the file
# foreach my $key (keys %ngramtypes) {
#     print $fh "$key\t$ngramtypes{$key}\n";
# }
#
# # Close the file
# close $fh;

computeentropy($infile);

sub computeentropy {
	my $file = $_[0];
	open( IN,  "<",  $file )    or die("can't open $file");
	open( OUT, ">",  $outfile ) or die("can't open $outfile");

	# current document is cached in @lines for second pass in computebits
	my $lc = 0;
	while ( defined( my $line = <IN> ) ) {
		if ( $line =~ /^<text id=\"([^\"]+)/ ) {    # new document
			print "$1\n";
			@lines          = ();
			$lc             = 0;
			$lines[$lc]     = $line;
			%docngrams      = ();
			%docngramtypes  = ();
			%restngramtypes = ();

			# insert a sentence marker at the beginning of a document
			@prevterms = ();
			addupdoc("SENT");
			@prevterms = ("SENT");
		}
		elsif ( $line =~ /^<\/text/ ) {
			$lines[$lc] = $line;
			$docngrams{"SENT"} = $docngrams{"SENT"} - 1;
			$docngrams{"\r"} = $docngrams{"\r"} - 1;

			# compute and output bits for this document;
			computebits();
		}
		elsif ( $line =~ /^</ ) {
			$lines[$lc] = $line;
		}
		else {
			$lines[$lc] = $line;
			my @fields = split( /\t/, $line );
			my $term   =  $fields[0];
			my $pos    = $fields[1];
			if ( $term =~ /\r/ ) {
				print "Warning: potential conflict with ngram encoding\n";
			}
			$term = lc($term);
			if ( $pos =~ /SENT/ ) {
				$term = "SENT";
			}
			elsif ( $terms{$term} == 1 ) {
				$term = "UNK";
			}
			if ( scalar(@prevterms) > $maxngram ) {
				shift(@prevterms);
			}
			addupdoc($term);
			push( @prevterms, $term );
			if ( $pos =~ /SENT/ ) {
				@prevterms = ("SENT");
			}
		}
		$lc++;
	}
	close OUT;
	close IN;
}

sub computebits {
	@prevterms = ("SENT");
	my $c        = 0;
	my $crossSum = 0;
	my $entSum   = 0;
	my $sc       = 0;
	%cacheterms  = ();
	$cachetokens = 0;
	%lastocc     = ();
	$cachecount  = 0;
	my $outbuffer     = "";
	my $sentbuffer    = "";
	my $sentencestart = 0;
	my $sentenceatts = "";

	for ( my $i = 0 ; $i < scalar(@lines) ; $i++ ) {
		my $line = $lines[$i];
		if ( $line =~ /^<text/ ) {
			$outbuffer = $outbuffer . $line;
		}
		elsif ( $line =~ /^<\/text/ ) {
			$outbuffer = $outbuffer . $line;
			print OUT $outbuffer;

			# end of this text;
			last;
		}
		elsif ( $line =~ /^<s(.*)>/ ) {
			$sentencestart = 1;
			$sentenceatts = $1;
		}
		elsif ( $line =~ /^<\/s>/ ) {
			my $crossAvg = 0;
			my $entAvg   = 0;
			if ( $c > 0 ) {
				$crossAvg = $crossSum / $c;
				$entAvg   = $entSum / $c;
			}
			if ($opt_nocross) {
				$outbuffer = $outbuffer
				. sprintf( "<s%s %s=\"%5.3f\">\n%s",
					 $sentenceatts, $ent_name, $entAvg, $sentbuffer );

			}
			elsif ($opt_noent) {
				$outbuffer = $outbuffer
				. sprintf( "<s%s %s=\"%5.3f\">\n%s",
					 $sentenceatts, $cross_name, $crossAvg, $sentbuffer );

			}
			else{
				$outbuffer = $outbuffer
				. sprintf( "<s%s %s=\"%5.3f\" %s=\"%5.3f\">\n%s",
					 $sentenceatts, $cross_name, $crossAvg, $ent_name, $entAvg, $sentbuffer );
			}
			$crossSum      = 0;
			$entSum        = 0;
			$c             = 0;
			$outbuffer     = $outbuffer . $line;
			$sentbuffer    = "";
			$sentencestart = 0;

		}
		elsif ( $line =~ /^</ ) {
			if ( $sentencestart == 1 ) {
				$sentbuffer = $sentbuffer . $line;
			}
			else {
				$outbuffer = $outbuffer . $line;
			}
		}
		else {
			my @fields = split( /\t/, $line );
			chomp($line);  # remove the trailing newline character (\n)
			my $term  =  $fields[0];
			my $pos   = $fields[1];
			my $term1 = lc($term);
			if ( $terms{$term1} == 1 ) {
				$term1 = "UNK";
			}
			elsif ( $pos =~ /SENT/ ) {
				$term1 = "SENT";
			}
			if ( scalar(@prevterms) > $maxngram ) {
				shift(@prevterms);
			}
			my $crossBits = getCrossBits($term1);
			my $entBits   = getEntBits($term1);

			if ($opt_nocross) {
				$sentbuffer =
				$sentbuffer . $line
				. sprintf( "\t%4.2f\n", $entBits );
			}
			elsif ($opt_noent) {
				$sentbuffer =
				$sentbuffer . $line
				. sprintf( "\t%4.2f\n", $crossBits );
			}
			else {
				$sentbuffer =
				$sentbuffer . $line
				. sprintf( "\t%4.2f\t%4.2f\n", $crossBits, $entBits );
			}
			$c++;
			$crossSum += $crossBits;
			$entSum   += $entBits;

			push( @prevterms, $term1 );
			if ( $term1 eq "SENT" ) {
				@prevterms = ("SENT");
			}
			$cachetokens = $tau * $cachetokens + 1;
			if ( exists $cacheterms{$term1} ) {
				$cacheterms{$term1} =
				  ( $tau**( $cachecount - $lastocc{$term1} ) ) *
				  $cacheterms{$term1} + 1;
			}
			else {
				$cacheterms{$term1} = 1;
			}
			$lastocc{$term1} = $cachecount;
			$cachecount++;
		}
	}
}

sub collectterms {
	my $file = $_[0];
	open( IN, "<",  $file ) or die("can't open $file");
	while ( defined( my $line = <IN> ) ) {
		if ( $line =~ /^</ ) { }
		else {
			my @fields = split( /\t/, $line );
			my $term   =  $fields[0];
			my $pos    = $fields[1];
			$term = lc($term);
			if ( $pos =~ /SENT/ ) {
				$term = "SENT";
			}
			if ( exists $terms{$term} ) {
				$terms{$term} = $terms{$term} + 1;
			}
			else { $terms{$term} = 1; }
		}
	}
	close(IN);
}

sub collectngrams {
	my $file = $_[0];
	open( IN, "<", $file ) or die("can't open $file");
	while ( defined( my $line = <IN> ) ) {
		if ( $line =~ /^<text/ ) {    # new document
				# insert a sentence marker at the beginning of a document
			@prevterms = ();
			addup("SENT");
			@prevterms = ("SENT");
		}
		elsif ( $line =~ /^<\/text/ ) {

			# subtract last sentence marker to compensate for the extra sentence marker at the beginning of text.
			# todo: is this necessary?
			$ngrams{"SENT"} = $ngrams{"SENT"} - 1;
			$ngrams{"\r"} = $ngrams{"\r"} - 1;
		}
		elsif ( $line =~ /^</ ) { }
		else {
			my @fields = split( /\t/, $line );
			my $term   =  $fields[0];
			my $pos    = $fields[1];
			$term = lc($term);
			if ( $pos =~ /SENT/ ) {
				$term = "SENT";
			}
			elsif ( $terms{$term} == 1 ) {
				$term = "UNK";
			}
			if ( scalar(@prevterms) > $maxngram ) {
				shift(@prevterms);
			}
			addup($term);
			push( @prevterms, $term );
			if ( $pos =~ /SENT/ ) {
				@prevterms = ("SENT");
			}
		}
	}
	close(IN);
}

sub addup {
	my $term  = shift;
	my $ngram = $term;

	# Unigram probability of $term
	if ( exists $ngrams{$ngram} ) {
		$ngrams{$ngram} = $ngrams{$ngram} + 1;
	}
	else {
		$ngrams{$ngram} = 1;

		# number of unigram types
		if ( exists $ngramtypes{"\r"} ) {
			$ngramtypes{"\r"} = $ngramtypes{"\r"} + 1;
		}
		else {
			$ngramtypes{"\r"} = 1;
		}
	}

	# number of unigram tokens
	if ( exists $ngrams{"\r"} ) {
		$ngrams{"\r"} = $ngrams{"\r"} + 1;
	}
	else { $ngrams{"\r"} = 1; }

	# same for n-grams
	for ( my $i = scalar(@prevterms) ; $i > 0 ; $i-- ) {
		$ngram = $prevterms[ $i - 1 ] . "\r" . $ngram;
		my $context = $ngram;
		$context =~ s/\r[^\r]+$//;
		if ( exists $ngrams{$ngram} ) {
			$ngrams{$ngram} = $ngrams{$ngram} + 1;
		}
		else {
			$ngrams{$ngram} = 1;
			if ( exists $ngramtypes{$context} ) {
				$ngramtypes{$context} = $ngramtypes{$context} + 1;
			}
			else {
				$ngramtypes{$context} = 1;
			}
		}
	}
}

sub addupdoc {
	my $term  = shift;
	my $ngram = $term;
	if ( exists $docngrams{$ngram} ) {
		$docngrams{$ngram} = $docngrams{$ngram} + 1;
	}
	else {
		$docngrams{$ngram} = 1;

		# number of unigram types
		if ( exists $docngramtypes{"\r"} ) {
			$docngramtypes{"\r"} = $docngramtypes{"\r"} + 1;
		}
		else {
			$docngramtypes{"\r"} = 1;
		}
	}
	if ( not( exists $restngramtypes{"\r"} ) ) {
		$restngramtypes{"\r"} = $ngramtypes{"\r"};
	}

	# ngram only occurs in this document
	if ( $ngrams{$ngram} - $docngrams{$ngram} == 0 ) {
		$restngramtypes{"\r"} = $restngramtypes{"\r"} - 1;
	}
	if ( exists $docngrams{"\r"} ) {
		$docngrams{"\r"} = $docngrams{"\r"} + 1;
	}
	else { $docngrams{"\r"} = 1; }
	for ( my $i = scalar(@prevterms) ; $i > 0 ; $i-- ) {
		$ngram = $prevterms[ $i - 1 ] . "\r" . $ngram;
		my $context = $ngram;
		# This pattern matches all the characters at the end of the string that follow a carriage return, excluding the carriage return itself.
		$context =~ s/\r[^\r]+$//;
		if ( exists $docngrams{$ngram} ) {
			$docngrams{$ngram} = $docngrams{$ngram} + 1;
		}
		else {
			$docngrams{$ngram} = 1;
			if ( exists $docngramtypes{$context} ) {
				$docngramtypes{$context} = $docngramtypes{$context} + 1;
			}
			else {
				$docngramtypes{$context} = 1;
			}
		}
		if ( not( exists $restngramtypes{$context} ) ) {
			$restngramtypes{$context} = $ngramtypes{$context};
		}
		if ( $ngrams{$ngram} - $docngrams{$ngram} == 0 ) {
			$restngramtypes{$context} = $restngramtypes{$context} - 1;
		}
	}
}

sub getCrossBits {
	my $term  = shift;
	my $ngram = $term;
	for ( my $i = scalar(@prevterms) ; $i > 0 ; $i-- ) {
		$ngram = $prevterms[ $i - 1 ] . "\r" . $ngram;
	}
	my $prob = getWBcross( $ngram, 0 );
	if ( $cachetokens > 0 ) {
	# the backoff unigram model in (1-$gamma) also exludes the current document,
	# probably not a big deal.
		my $cacheprob = $gamma * ( $tau**( $cachecount - $lastocc{$term} ) ) *
		  $cacheterms{$term} / $cachetokens + ( 1 - $gamma ) *
		( ( $ngrams{$term} - $docngrams{$term} ) ) /
		( ( $ngrams{"\r"} - $docngrams{"\r"} ) );
		$prob = $lambda * $prob + ( 1 - $lambda ) * $cacheprob;
	}
	if ( $prob > 1 ) {
		my $context = $ngram;
		$context =~ s/\r[^\r]+$//;
		die(
"$ngram,$prob,$docngrams{$ngram},$restngramtypes{$context},$docngrams{$term},$restngramtypes{'\r'}\n"
		);
	}
	return sprintf( "%5.3f", -log2($prob) );
}

sub getEntBits {
	my $term  = shift;
	my $ngram = $term;
	for ( my $i = scalar(@prevterms) ; $i > 0 ; $i-- ) {
		$ngram = $prevterms[ $i - 1 ] . "\r" . $ngram;
	}
	return -log2( getWBent($ngram) );
}

sub getWBcross {
	my $ngram = shift;
	my $count = shift;

	if ( $count > 8 ) {
		die("what? $ngram\n");
	}
	if ( $ngram eq '' ) {
		return 1.0 / $restngramtypes{"\r"};
	}
	my $context = $ngram;
	my $rest    = $ngram;
	if ( $context =~ /\r/ ) {
		$context =~ s/\r[^\r]+$//;
		$rest    =~ s/^[^\r]+\r//;
	}
	else {
		$context = "\r";
		$rest    = '';
	}

	my $typecount = $restngramtypes{$context};
	if ( $ngrams{$ngram} - $docngrams{$ngram} == 0 ) {
		return getWBcross($rest);
	}
	my $mle =
	  ( $ngrams{$ngram} - $docngrams{$ngram} ) /
	  ( $ngrams{$context} - $docngrams{$context} );
	my $lambda1 =
	  ( $ngrams{$context} - $docngrams{$context} ) /
	  ( $ngrams{$context} - $docngrams{$context} + $typecount );
	$sumlambda += $lambda1;
	$countlambda++;
	return $lambda1 * $mle + ( 1 - $lambda1 ) * getWBcross( $rest, $count + 1 );
}

sub getWBent {
	my $ngram = shift;
	my $count = shift;

	if ( $count > 8 ) {
		die("what? $ngram\n");
	}
	if ( $ngram eq '' ) {
		return 1.0 / $docngramtypes{"\r"};
	}
	my $context = $ngram;
	my $rest    = $ngram;
	if ( $context =~ /\r/ ) {
		$context =~ s/\r[^\r]+$//;
		$rest    =~ s/^[^\r]+\r//;
	}
	else {
		$context = "\r";
		$rest    = '';
	}

	my $typecount = $docngramtypes{$context};
	if ( $docngrams{$ngram} == 0 ) {
		return getWBent($rest);
	}
	my $mle = $docngrams{$ngram} / $docngrams{$context};
	my $lambda1 = $docngrams{$context} / ( $docngrams{$context} + $typecount );
	return $lambda1 * $mle + ( 1 - $lambda1 ) * getWBent( $rest, $count + 1 );
}

sub log2 {
	my $n = shift;
	return log($n) / log(2);
}

