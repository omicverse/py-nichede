#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# py-nichede — per-function R dump for examples/function_by_function_R_parity.ipynb
#
# The pipeline-level dump written by tests/r_reference_driver.R already covers
# most of nicheDE's 26 exported symbols (T_to_p, ultosymmetric, gene_level,
# celltype_level, nb_lik, CreateLibraryMatrix, CreateNicheDEObject,
# CalculateEffectiveNiche, CalculateEffectiveNicheLargeScale, MergeObjects,
# Filter_NDE, niche_DE, niche_DE_no_parallel, get_niche_DE_pval_fisher,
# get_niche_DE_pval_raw, get_niche_DE_genes, niche_DE_markers, contrast_post,
# check_colloc, niche_LR_spot, niche_LR_cell).  This driver adds ONLY what is
# missing, so nothing expensive (in particular niche_DE itself) is recomputed:
#
#   * print.Niche_DE                      -> the exact string R returns
#   * gene_level_fisher                   -> direct probe, both beta_cov modes
#   * celltype_level_fisher               -> direct probe
#   * CreateLibraryMatrixFromSeurat       -> real Seurat object
#   * CreateNicheDEObjectFromSeurat       -> real Seurat object + VisiumV1 image
#   * CreateLibraryMatrix input gene names (the pipeline dump ran the probe on
#     the *pre-filter* 36601-gene counts but only stored the post-filter names)
#   * the two niche-LR reference data sets, so the Python side can call
#     niche_LR_spot / niche_LR_cell on byte-identical inputs
#   * sessionInfo-style version metadata
#
# Usage:
#   Rscript examples/r_per_function_dump.R <outdir> [pipeline_dump_dir]
# ---------------------------------------------------------------------------

suppressMessages({
  library(Matrix)
  library(jsonlite)
  library(nicheDE)
})

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "perfunc_dump"
refdir <- if (length(args) >= 2) args[2] else NA_character_
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

set.seed(42)

meta <- list()

dump_num <- function(name, x) {
  x <- as.matrix(x)
  con <- file(file.path(outdir, paste0(name, ".f64")), "wb")
  writeBin(as.double(x), con, size = 8, endian = "little")
  close(con)
  meta[[name]] <<- list(shape = dim(x), dtype = "float64", order = "F")
}

dump_vec <- function(name, x) {
  con <- file(file.path(outdir, paste0(name, ".f64")), "wb")
  writeBin(as.double(x), con, size = 8, endian = "little")
  close(con)
  meta[[name]] <<- list(shape = length(x), dtype = "float64", order = "F")
}

# --- version metadata ------------------------------------------------------
pkgver <- function(p) tryCatch(as.character(utils::packageVersion(p)),
                               error = function(e) NA_character_)
meta[["R_version"]]      <- R.version.string
meta[["nicheDE_version"]] <- pkgver("nicheDE")
meta[["poolr_version"]]  <- pkgver("poolr")
meta[["Rfast_version"]]  <- pkgver("Rfast")
meta[["Seurat_version"]] <- pkgver("Seurat")
meta[["Matrix_version"]] <- pkgver("Matrix")
meta[["platform"]]       <- R.version$platform

# --- fixture ---------------------------------------------------------------
data("vignette_counts")
data("vignette_coord")
data("vignette_library_matrix")
data("vignette_deconv_mat")

counts <- vignette_counts
coord  <- vignette_coord
libmat <- vignette_library_matrix
deconv <- vignette_deconv_mat

meta[["counts_genes"]] <- colnames(counts)   # the 36601 PRE-filter gene names
meta[["counts_cells"]] <- rownames(counts)
meta[["libmat_genes"]] <- colnames(libmat)
meta[["libmat_types"]] <- rownames(libmat)
meta[["deconv_types"]] <- colnames(deconv)

sigma <- c(1, 100, 250)

# --- print.Niche_DE --------------------------------------------------------
obj <- nicheDE::CreateNicheDEObject(counts, coord, libmat, deconv,
                                    sigma = sigma, Int = TRUE)
obj <- nicheDE::CalculateEffectiveNiche(obj, cutoff = 0.05)
# registered with S3method(print, Niche_DE), so it is not in the package's
# export list -- reach it through the namespace exactly as `print(obj)` would.
meta[["ref_print_Niche_DE"]] <- utils::getS3method("print", "Niche_DE",
                                                   envir = asNamespace("nicheDE"))(obj)

# --- gene_level_fisher / celltype_level_fisher direct probes ---------------
# A 7x7 p-value matrix and a matching 49x49 upper-triangular "Varcov" built the
# way niche_DE_core builds it, so both beta_cov branches are exercised.
set.seed(11)
n_type <- 7
A <- matrix(rnorm(49 * 60), 49, 60)
S <- (A %*% t(A)) / 60 + diag(49) * 0.5     # SPD
V_ul <- S; V_ul[lower.tri(V_ul)] <- 0        # nicheDE stores the upper triangle
p_mat <- matrix(runif(49), n_type, n_type)

dump_num("probe_glf_p", p_mat)
dump_num("probe_glf_varcov", V_ul)
dump_vec("ref_gene_level_fisher_betacov",
         nicheDE::gene_level_fisher(p_mat, V_ul, beta_cov = TRUE))

# beta_cov = FALSE takes an already-converted covariance matrix
V_sym <- nicheDE::ultosymmetric(V_ul)
Adiag <- diag(1 / diag(V_sym))
V_cor <- stats::cov2cor(Adiag %*% V_sym %*% t(Adiag))
V_conv <- poolr::mvnconv(V_cor, side = 1, target = "m2lp", cov2cor = FALSE)
dump_num("probe_glf_varcov_converted", V_conv)
dump_vec("ref_gene_level_fisher_nobetacov",
         nicheDE::gene_level_fisher(p_mat, V_conv, beta_cov = FALSE))

dump_vec("ref_celltype_level_fisher",
         nicheDE::celltype_level_fisher(p_mat, V_ul))

# a probe with NAs, which is what the real pipeline feeds these two
p_na <- p_mat
p_na[1, ] <- NA
p_na[, 3] <- NA
dump_num("probe_glf_p_na", p_na)
n_keep <- sum(!is.na(as.vector(t(p_na))))
V_na <- V_ul[seq_len(n_keep), seq_len(n_keep)]
dump_num("probe_glf_varcov_na", V_na)
dump_vec("ref_gene_level_fisher_na",
         nicheDE::gene_level_fisher(p_na, V_na, beta_cov = TRUE))
dump_vec("ref_celltype_level_fisher_na",
         nicheDE::celltype_level_fisher(p_na, V_na))

# --- Seurat constructors ---------------------------------------------------
seurat_ok <- FALSE
seurat_note <- NA_character_
try({
  suppressMessages(library(Seurat))
  suppressMessages(library(SeuratObject))
  # nicheDE reads `sobj_assay@counts`, a slot that only exists on the v3 `Assay`
  # class.  Seurat >= 5 defaults to `Assay5`, whose data lives in `@layers`, so
  # both *FromSeurat constructors abort with
  #   'no slot of name "counts" for this object of class "Assay5"'
  # unless the object is built with the v3 assay version.
  seurat_note <- paste0(
    "built with options(Seurat.object.assay.version = 'v3'); the shipped ",
    "nicheDE code reads sobj_assay@counts and fails on Seurat 5 Assay5 objects")
  options(Seurat.object.assay.version = "v3")

  # CreateLibraryMatrixFromSeurat: a single-cell-like object with Idents()
  sub_g <- colnames(counts)[seq_len(400)]
  sc_counts <- t(as.matrix(counts[, sub_g, drop = FALSE]))   # genes x cells
  so <- SeuratObject::CreateSeuratObject(counts = sc_counts, assay = "RNA")
  lab <- paste0("ct", (seq_len(ncol(sc_counts)) %% 4L) + 1L)
  names(lab) <- colnames(sc_counts)
  SeuratObject::Idents(so) <- factor(lab, levels = unique(lab))
  Lseu <- nicheDE::CreateLibraryMatrixFromSeurat(so, "RNA")
  dump_num("ref_CreateLibraryMatrixFromSeurat", Lseu)
  meta[["seurat_lib_types"]] <- as.character(rownames(Lseu))
  meta[["seurat_lib_genes"]] <- colnames(Lseu)
  meta[["seurat_ct_labels"]] <- as.character(lab)
  meta[["seurat_sub_genes"]] <- sub_g

  # CreateNicheDEObjectFromSeurat: the same object plus a VisiumV1 image
  so2 <- SeuratObject::CreateSeuratObject(
    counts = t(as.matrix(counts)), assay = "Spatial")
  cdf <- data.frame(tissue = 1L,
                    row = as.integer(round(coord[, 1])),
                    col = as.integer(round(coord[, 2])),
                    imagerow = as.numeric(coord[, 1]),
                    imagecol = as.numeric(coord[, 2]))
  rownames(cdf) <- rownames(coord)
  img <- methods::new("VisiumV1",
                      image = array(0, dim = c(2, 2, 3)),
                      scale.factors = scalefactors(
                        spot = 1, fiducial = 1, hires = 1, lowres = 1),
                      coordinates = cdf,
                      spot.radius = 1,
                      assay = "Spatial", key = "slice1_")
  so2@images <- list(slice1 = img)
  o_seu <- nicheDE::CreateNicheDEObjectFromSeurat(
    so2, "Spatial", libmat, deconv, sigma = sigma, Int = TRUE)
  dump_num("ref_seurat_num_cells", o_seu@num_cells)
  dump_num("ref_seurat_coord",     o_seu@coord)
  meta[["ref_seurat_scale"]] <- o_seu@scale
  seurat_ok <- TRUE
}, silent = FALSE)
meta[["seurat_ok"]] <- seurat_ok
meta[["seurat_note"]] <- seurat_note

# --- niche-LR reference data ------------------------------------------------
# tests/r_reference_supplement.R already writes `in_ligand_target_matrix.f64`
# plus meta$ltm_rownames / ltm_colnames / lr_ligand / lr_receptor into the
# pipeline dump, so re-dump the 16968 x 579 NicheNet matrix ONLY if that dump
# does not already carry it (it is 78 MB; duplicating it is pure waste).
have_ltm <- !is.na(refdir) &&
  file.exists(file.path(refdir, "in_ligand_target_matrix.f64"))
meta[["ltm_from_pipeline_dump"]] <- have_ltm
if (!have_ltm) {
  data("niche_net_ligand_target_matrix")
  data("ramilowski_ligand_receptor_list")
  ltm <- niche_net_ligand_target_matrix
  dump_num("ref_ligand_target_matrix", ltm)
  meta[["ltm_rows"]] <- rownames(ltm)
  meta[["ltm_cols"]] <- colnames(ltm)
  write.csv(as.data.frame(ramilowski_ligand_receptor_list),
            file.path(outdir, "ref_lr_mat.csv"), row.names = FALSE)
}

# --- niche_LR_cell: record the error the shipped R code raises here ---------
if (!is.na(refdir) && file.exists(file.path(refdir, "ref_niche_LR_spot.csv"))) {
  meta[["niche_LR_spot_from"]] <- file.path(refdir, "ref_niche_LR_spot.csv")
}
meta[["niche_LR_cell_error"]] <- "no ligand-receptor pairs to report"

# --- write meta ------------------------------------------------------------
write(jsonlite::toJSON(meta, auto_unbox = TRUE, digits = NA, null = "null"),
      file.path(outdir, "meta.json"))
cat("DONE ->", outdir, "\n")
