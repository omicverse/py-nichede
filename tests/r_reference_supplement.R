#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Supplementary reference dump: the two pieces `r_reference_driver.R` cannot
# emit inline — the ligand/receptor resources needed to parity-test niche-LR,
# and a CreateLibraryMatrix probe run on the *gene-filtered* counts so the
# Python side can feed it byte-identical input.
#
#   Rscript tests/r_reference_supplement.R <outdir> [n_genes]
# ---------------------------------------------------------------------------

suppressMessages({
  library(Matrix); library(jsonlite); library(nicheDE)
})

args <- commandArgs(trailingOnly = TRUE)
outdir  <- args[1]
n_genes <- if (length(args) >= 2) as.integer(args[2]) else 0L

meta <- fromJSON(file.path(outdir, "meta.json"))

dump_num <- function(name, x) {
  x <- as.matrix(x)
  con <- file(file.path(outdir, paste0(name, ".f64")), "wb")
  writeBin(as.double(x), con, size = 8, endian = "little"); close(con)
  meta[[name]] <<- list(shape = dim(x), dtype = "float64", order = "F")
}

data("vignette_counts"); data("vignette_coord")
data("vignette_library_matrix"); data("vignette_deconv_mat")
counts <- vignette_counts; libmat <- vignette_library_matrix
if (n_genes > 0L) {
  shared <- intersect(colnames(libmat), colnames(counts))
  tot    <- colSums(counts[, shared, drop = FALSE])
  keep   <- names(sort(tot, decreasing = TRUE))[seq_len(min(n_genes, length(tot)))]
  keep   <- colnames(libmat)[colnames(libmat) %in% keep]
  counts <- counts[, keep, drop = FALSE]; libmat <- libmat[, keep, drop = FALSE]
}
obj <- suppressMessages(nicheDE::CreateNicheDEObject(
  counts, vignette_coord, libmat, vignette_deconv_mat,
  sigma = c(1, 100, 250), Int = TRUE))
filtered <- as.matrix(obj@counts)

# --- CreateLibraryMatrix on the gene-filtered counts ------------------------
probe_ct <- data.frame(cell = rownames(filtered),
                       type = paste0("ct", (seq_len(nrow(filtered)) %% 4L) + 1L),
                       stringsAsFactors = FALSE)
L <- nicheDE::CreateLibraryMatrix(filtered, probe_ct)
dump_num("ref_CreateLibraryMatrix", L)
meta[["probe_ct_types"]]  <- rownames(L)
meta[["probe_ct_labels"]] <- probe_ct$type

# --- niche-LR resources -----------------------------------------------------
data("niche_net_ligand_target_matrix")
data("ramilowski_ligand_receptor_list")
ltm <- niche_net_ligand_target_matrix
dump_num("in_ligand_target_matrix", ltm)
meta[["ltm_rownames"]] <- rownames(ltm)
meta[["ltm_colnames"]] <- colnames(ltm)
lr <- ramilowski_ligand_receptor_list
meta[["lr_ligand"]]   <- as.character(lr[, 1])
meta[["lr_receptor"]] <- as.character(lr[, 2])

# --- multi-batch niche_DE anchor --------------------------------------------
# The canonical fixture is a single section, so the multi-batch code path
# (factor `batchvar` -> fastDummies treatment contrasts in the GLM, plus
# MergeObjects' coordinate renormalisation) would otherwise never be gated.
# Merge the fixture with a 1.5x-rescaled copy of itself to get two batches.
dump_vec <- function(name, x) {
  con <- file(file.path(outdir, paste0(name, ".f64")), "wb")
  writeBin(as.double(x), con, size = 8, endian = "little"); close(con)
  meta[[name]] <<- list(shape = length(x), dtype = "float64", order = "F")
}
n_mb <- min(300L, ncol(counts))
tot_mb <- colSums(counts)
keep_mb <- names(sort(tot_mb, decreasing = TRUE))[seq_len(n_mb)]
keep_mb <- colnames(libmat)[colnames(libmat) %in% keep_mb]
c_mb <- counts[, keep_mb, drop = FALSE]; l_mb <- libmat[, keep_mb, drop = FALSE]
o1 <- suppressMessages(nicheDE::CreateNicheDEObject(
  c_mb, vignette_coord, l_mb, vignette_deconv_mat, sigma = c(1, 100, 250)))
o2 <- suppressMessages(nicheDE::CreateNicheDEObject(
  c_mb, vignette_coord * 1.5, l_mb, vignette_deconv_mat, sigma = c(1, 100, 250)))
om <- nicheDE::MergeObjects(list(o1, o2))
om <- nicheDE::CalculateEffectiveNiche(om, cutoff = 0.05)
om <- nicheDE::niche_DE_no_parallel(om, C = 150, M = 10, gamma = 0.8,
                                    print = FALSE, Int = TRUE, batch = TRUE,
                                    self_EN = FALSE)
nct <- length(om@cell_types); ngm <- length(om@gene_names)
for (k in seq_along(om@sigma)) {
  Ts <- array(NA_real_, dim = c(nct, nct, ngm)); vd <- rep(0, ngm)
  for (g in seq_len(ngm)) {
    r <- om@niche_DE[[k]][[g]]
    if (isTRUE(r$valid == 1)) Ts[, , g] <- as.matrix(r$T_stat)
    vd[g] <- as.numeric(r$valid)
  }
  dump_num(paste0("ref_mb_T_stat_", k), matrix(Ts, nct * nct, ngm))
  dump_vec(paste0("ref_mb_valid_", k), vd)
  dump_num(paste0("ref_mb_en_", k), om@effective_niche[[k]])
}
dump_num("ref_mb_num_cells", om@num_cells)
dump_num("ref_mb_coord", om@coord)
dump_vec("ref_mb_batch", om@batch_ID)
dump_vec("ref_mb_pval_pos_gene", om@niche_DE_pval_pos$gene_level)
meta[["mb_genes"]] <- om@gene_names

write(toJSON(meta, auto_unbox = TRUE, digits = NA, null = "null"),
      file.path(outdir, "meta.json"))
cat("SUPPLEMENT DONE ->", outdir, "\n")
