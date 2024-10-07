.libPaths("~/Rlibs")

library(GenomicRanges)
library(GenomicFeatures)
library(rtracklayer)
library(dplyr)
library(Biostrings)

gtf <- '/fast/AG_Ohler/frederick/projects/mpra/library_design/annotations/human_gencode/gencode.v44.primary_assembly.annotation.gtf'
fafile <- '/fast/AG_Ohler/frederick/projects/mpra/library_design/annotations/human_gencode/GRCh38.p14.genome.fa'
fputrfile <- '/fast/AG_Ohler/frederick/projects/mpra/library_design/seqs/gencode_ccds_fputrs_human.fa'

stopifnot(file.exists(fafile))
stopifnot(file.exists(gtf))
fafileob = Rsamtools::FaFile(fafile)
Rsamtools::indexFa(fafile)

#get utrs
#load the gtf as a granges
gtf_gr <- rtracklayer::import(con=gtf,format='gtf')

#export ccds entries as gtf file
rtracklayer::export(gtf_gr, '/fast/AG_Ohler/frederick/projects/mpra/library_design/annotations/human_gencode/gencode.v44.primary_assembly.annotation.ccds.gtf')

#create translation table with transcript_id to gene_id and ccds_id
txid2ids <- gtf_ccds_gr %>%
	mcols() %>%
	as.data.frame() %>%
	distinct(transcript_id, gene_id, ccdsid) %>%
	select(transcript_id, gene_id, ccdsid)

#export table
write.csv(txid2ids, file = "/fast/AG_Ohler/frederick/projects/mpra/library_design/annotations/human_gencode/txid2ids.csv", row.names = FALSE)

#make a transcript db object
gtftxdb <- makeTxDbFromGRanges(gtf_ccds_gr)

#get utrs from this as grangeslists
fputrs <- fiveUTRsByTranscript(gtftxdb, use.names=TRUE)

#extract sequence and print to a file
GenomicFeatures::extractTranscriptSeqs(fputrs,x=fafileob) %>% writeXStringSet(fputrfile)



