#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# py-nichede — R reference runner (Omicverse-RebuildR PROTOCOL.md Step 3.3)
#
# Runs the upstream nicheDE R package end-to-end on the canonical fixture
# (the vignette data shipped inside nicheDE itself) and dumps BOTH the inputs
# and every reference output as flat little-endian float64 / int32 binaries
# plus a JSON sidecar describing shapes and names.
#
# Usage:
#   Rscript tests/r_reference_driver.R <outdir> [n_genes] [n_cores]
#
#   n_genes = 0  -> use every gene shared between counts and the library matrix
#                   (the canonical full fixture)
#   n_genes = K  -> use the K shared genes with the largest total count
#                   (deterministic dev fixture; the same K genes are handed to
#                    the Python side so gamma-quantile filters agree)
# ---------------------------------------------------------------------------

suppressMessages({
  library(Matrix)
  library(jsonlite)
  library(nicheDE)
})

args <- commandArgs(trailingOnly = TRUE)
outdir  <- args[1]
n_genes <- if (length(args) >= 2) as.integer(args[2]) else 0L
n_cores <- if (length(args) >= 3) as.integer(args[3]) else 4L
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

set.seed(42)

# --- helpers ---------------------------------------------------------------
meta <- list()

dump_num <- function(name, x) {
  x <- as.matrix(x)
  con <- file(file.path(outdir, paste0(name, ".f64")), "wb")
  # column-major (R native); Python reads with order="F"
  writeBin(as.double(x), con, size = 8, endian = "little")
  close(con)
  meta[[name]] <<- list(shape = dim(x), dtype = "float64", order = "F")
}

dump_arr3 <- function(name, x) {
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

# --- fixture ---------------------------------------------------------------
data("vignette_counts")
data("vignette_coord")
data("vignette_library_matrix")
data("vignette_deconv_mat")

counts  <- vignette_counts
coord   <- vignette_coord
libmat  <- vignette_library_matrix
deconv  <- vignette_deconv_mat

if (n_genes > 0L) {
  shared <- intersect(colnames(libmat), colnames(counts))
  tot    <- colSums(counts[, shared, drop = FALSE])
  keep   <- names(sort(tot, decreasing = TRUE))[seq_len(min(n_genes, length(tot)))]
  keep   <- colnames(libmat)[colnames(libmat) %in% keep]   # keep libmat order
  counts <- counts[, keep, drop = FALSE]
  libmat <- libmat[, keep, drop = FALSE]
}

cat("fixture:", dim(counts), "counts /", dim(libmat), "libmat\n")

# --- inputs (so Python starts from byte-identical data) --------------------
dump_num("in_counts",  counts)
dump_num("in_coord",   as.matrix(coord))
dump_num("in_libmat",  libmat)
dump_num("in_deconv",  deconv)

# --- CreateLibraryMatrix parity probe --------------------------------------
# Build a tiny single-cell-like matrix out of the spatial counts and a
# deterministic cell-type label so create_library_matrix can be parity-tested
# without needing a separate reference dataset.
probe_ct <- data.frame(
  cell = rownames(counts),
  type = paste0("ct", (seq_len(nrow(counts)) %% 4L) + 1L),
  stringsAsFactors = FALSE
)
probe_L <- nicheDE::CreateLibraryMatrix(counts, probe_ct)
dump_num("ref_CreateLibraryMatrix", probe_L)
meta[["probe_ct_types"]] <- rownames(probe_L)
meta[["probe_ct_labels"]] <- probe_ct$type

# --- Step 1: CreateNicheDEObject -------------------------------------------
sigma <- c(1, 100, 250)
t0 <- Sys.time()
obj <- nicheDE::CreateNicheDEObject(counts, coord, libmat, deconv,
                                    sigma = sigma, Int = TRUE)
cat("CreateNicheDEObject:", as.numeric(Sys.time() - t0, units = "secs"), "s\n")

dump_num("ref_num_cells", obj@num_cells)
dump_num("ref_coord",     obj@coord)
dump_num("ref_ref_expr",  obj@ref_expr)
dump_num("ref_counts",    as.matrix(obj@counts))
meta[["ref_scale"]]         <- obj@scale
meta[["ref_spot_distance"]] <- obj@spot_distance
meta[["sigma"]]             <- obj@sigma
meta[["cell_types"]]        <- obj@cell_types
meta[["gene_names"]]        <- obj@gene_names
meta[["cell_names"]]        <- obj@cell_names

# --- Step 2: CalculateEffectiveNiche ---------------------------------------
t0 <- Sys.time()
obj <- nicheDE::CalculateEffectiveNiche(obj, cutoff = 0.05)
t_en <- as.numeric(Sys.time() - t0, units = "secs")
cat("CalculateEffectiveNiche:", t_en, "s\n")
meta[["time_effective_niche"]] <- t_en
for (k in seq_along(obj@sigma)) {
  dump_num(paste0("ref_effective_niche_", k), obj@effective_niche[[k]])
}

# large-scale variant (separate object so it does not clobber the reference)
obj_ls <- nicheDE::CreateNicheDEObject(counts, coord, libmat, deconv,
                                       sigma = sigma, Int = TRUE)
obj_ls <- nicheDE::CalculateEffectiveNicheLargeScale(obj_ls, batch_size = 200,
                                                     cutoff = 0.05,
                                                     standardize = TRUE)
for (k in seq_along(obj_ls@sigma)) {
  dump_num(paste0("ref_effective_niche_ls_", k), obj_ls@effective_niche[[k]])
}
rm(obj_ls); gc()

# --- CalculateEffectiveNicheLargeScale, with Rfast::dista repaired ----------
# Rfast 2.1.5.2's dista() returns an all-zero matrix whenever nrow(xnew) >= 4,
# which silently turns the kernel weights into 1 and makes the shipped
# CalculateEffectiveNicheLargeScale disagree with CalculateEffectiveNiche.
# This block re-runs the *same* R algorithm with base-R distances so the
# Python port has a genuine R anchor for that code path.
ls_fixed <- function(coord, ncell, sigmas, batch_size, cutoff, standardize) {
  out <- vector("list", length(sigmas))
  coord <- as.matrix(coord); ncell <- as.matrix(ncell)
  for (si in seq_along(sigmas)) {
    sig <- sigmas[si]
    EN <- matrix(NA, nrow(coord), ncol(ncell))
    num_iter <- ceiling(nrow(coord) / batch_size)
    square_side <- ceiling(sqrt(num_iter))
    xr <- range(coord[, 1]); yr <- range(coord[, 2])
    xl <- xr[2] - xr[1]; yl <- yr[2] - yr[1]
    xb <- seq(xr[1], xr[2] + xl / (2 * square_side), length.out = square_side)
    yb <- seq(yr[1], yr[2] + yl / (2 * square_side), length.out = square_side)
    xi <- findInterval(coord[, 1], xb); yi <- findInterval(coord[, 2], yb)
    for (j in 1:square_side) for (k in 1:square_side) {
      inds <- which(xi == j & yi == k)
      if (length(inds) == 0) next
      xg <- range(coord[inds, 1]); yg <- range(coord[inds, 2])
      pad <- sig * sqrt(-log(cutoff))
      xg <- c(xg[1] - pad, xg[2] + pad); yg <- c(yg[1] - pad, yg[2] + pad)
      cand <- which(spatstat.utils::inside.range(coord[, 1], xg) &
                    spatstat.utils::inside.range(coord[, 2], yg))
      D <- as.matrix(dist(rbind(coord[inds, , drop = FALSE],
                                coord[cand, , drop = FALSE])))[
             seq_along(inds), length(inds) + seq_along(cand), drop = FALSE]
      D <- exp(-D^2 / sig^2); D[D < cutoff] <- 0
      EN[inds, ] <- D %*% ncell[cand, , drop = FALSE]
    }
    if (standardize)
      EN <- apply(EN, 2, function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE))
    EN[is.na(EN)] <- 0
    out[[si]] <- EN
  }
  out
}
lsf <- ls_fixed(obj@coord, obj@num_cells, obj@sigma, 200, 0.05, TRUE)
for (k in seq_along(obj@sigma)) dump_num(paste0("ref_effective_niche_lsfix_", k), lsf[[k]])
rm(lsf); gc()

# --- Step 3: niche_DE ------------------------------------------------------
t0 <- Sys.time()
obj <- nicheDE::niche_DE(obj, num_cores = n_cores, outfile = "",
                         C = 150, M = 10, gamma = 0.8, print = TRUE,
                         Int = TRUE, batch = TRUE, self_EN = FALSE, G = 1)
t_nde <- as.numeric(Sys.time() - t0, units = "secs")
cat("niche_DE:", t_nde, "s\n")
meta[["time_niche_DE"]] <- t_nde
meta[["n_cores"]] <- n_cores

nct   <- length(obj@cell_types)
ngene <- length(obj@gene_names)
nsig  <- length(obj@sigma)

# T-stat / beta / loglik / valid / nulls, per sigma
for (k in seq_len(nsig)) {
  Ts   <- array(NA_real_, dim = c(nct, nct, ngene))
  Bs   <- array(NA_real_, dim = c(nct, nct, ngene))
  ll   <- rep(NA_real_, ngene)
  vd   <- rep(0, ngene)
  nnul <- rep(NA_real_, ngene)
  for (g in seq_len(ngene)) {
    r <- obj@niche_DE[[k]][[g]]
    if (isTRUE(r$valid == 1)) {
      Ts[, , g] <- as.matrix(r$T_stat)
      Bs[, , g] <- as.matrix(r$betas)
    }
    if (!is.null(r$log_likelihood) && length(r$log_likelihood) == 1)
      ll[g] <- as.numeric(r$log_likelihood)
    vd[g]   <- as.numeric(r$valid)
    nnul[g] <- length(r$nulls)
  }
  dump_arr3(paste0("ref_T_stat_", k), Ts)
  dump_arr3(paste0("ref_betas_",  k), Bs)
  dump_vec(paste0("ref_loglik_",  k), ll)
  dump_vec(paste0("ref_valid_",   k), vd)
  dump_vec(paste0("ref_nnull_",   k), nnul)
}

# Varcov of the first 50 valid genes at sigma[1] (full dump would be huge)
vc_genes <- c(); vc_flat <- c(); vc_dim <- c()
for (g in seq_len(ngene)) {
  r <- obj@niche_DE[[1]][[g]]
  if (isTRUE(r$valid == 1) && length(vc_genes) < 50) {
    V <- as.matrix(r$Varcov)
    vc_genes <- c(vc_genes, g)
    vc_dim   <- c(vc_dim, nrow(V))
    vc_flat  <- c(vc_flat, as.double(V))
  }
}
dump_vec("ref_varcov_flat", vc_flat)
meta[["varcov_genes"]] <- vc_genes
meta[["varcov_dim"]]   <- vc_dim

# nulls of the first 200 genes at sigma[1]
nl_flat <- c(); nl_len <- c()
for (g in seq_len(min(200, ngene))) {
  nn <- obj@niche_DE[[1]][[g]]$nulls
  nl_len  <- c(nl_len, length(nn))
  nl_flat <- c(nl_flat, as.double(nn))
}
dump_vec("ref_nulls_flat", nl_flat)
meta[["nulls_len"]] <- nl_len

# --- Int = FALSE (linear-model) path, on the 300 highest-count genes --------
# The vignette fixture is integer counts, so the Int = FALSE branch is exercised
# on a log-transformed copy of the same data to give it a real R anchor.
n_small <- min(300, ncol(counts))
tot_s <- colSums(counts)
keep_s <- names(sort(tot_s, decreasing = TRUE))[seq_len(n_small)]
keep_s <- colnames(libmat)[colnames(libmat) %in% keep_s]
cont <- log1p(counts[, keep_s, drop = FALSE])
obj_c <- nicheDE::CreateNicheDEObject(cont, coord, log1p(libmat[, keep_s, drop = FALSE]),
                                      deconv, sigma = sigma, Int = FALSE)
obj_c <- nicheDE::CalculateEffectiveNiche(obj_c, cutoff = 0.05)
obj_c <- nicheDE::niche_DE_no_parallel(obj_c, C = 150, M = 10, gamma = 0.8,
                                       print = FALSE, Int = FALSE, batch = TRUE,
                                       self_EN = FALSE)
meta[["cont_genes"]] <- obj_c@gene_names
dump_num("in_cont_counts", as.matrix(obj_c@counts))
dump_num("in_cont_libmat", as.matrix(obj_c@ref_expr))
{
  ngc <- length(obj_c@gene_names)
  Tc <- array(NA_real_, dim = c(nct, nct, ngc)); vdc <- rep(0, ngc)
  for (g in seq_len(ngc)) {
    r <- obj_c@niche_DE[[1]][[g]]
    if (isTRUE(r$valid == 1)) Tc[, , g] <- as.matrix(r$T_stat)
    vdc[g] <- as.numeric(r$valid)
  }
  dump_arr3("ref_cont_T_stat_1", Tc)
  dump_vec("ref_cont_valid_1", vdc)
  dump_vec("ref_cont_pval_pos_gene", obj_c@niche_DE_pval_pos$gene_level)
}
rm(obj_c); gc()

# --- p-values --------------------------------------------------------------
dump_vec("ref_pval_pos_gene", obj@niche_DE_pval_pos$gene_level)
dump_num("ref_pval_pos_ct",   obj@niche_DE_pval_pos$cell_type_level)
dump_arr3("ref_pval_pos_int", obj@niche_DE_pval_pos$interaction_level)
dump_vec("ref_pval_neg_gene", obj@niche_DE_pval_neg$gene_level)
dump_num("ref_pval_neg_ct",   obj@niche_DE_pval_neg$cell_type_level)
dump_arr3("ref_pval_neg_int", obj@niche_DE_pval_neg$interaction_level)

# raw (un-BH-adjusted) p-values via get_niche_DE_pval_raw
obj_raw <- nicheDE::get_niche_DE_pval_raw(obj, pos = TRUE)
dump_vec("ref_praw_pos_gene", obj_raw@niche_DE_pval_pos$gene_level)
dump_num("ref_praw_pos_ct",   obj_raw@niche_DE_pval_pos$cell_type_level)
dump_arr3("ref_praw_pos_int", obj_raw@niche_DE_pval_pos$interaction_level)
rm(obj_raw); gc()

# --- Step 4: downstream ----------------------------------------------------
index_ct <- "tumor_epithelial"
niche_ct <- "myeloid"
niche_ct2 <- "stromal"

wr_df <- function(name, df) {
  write.csv(df, file.path(outdir, paste0(name, ".csv")), row.names = FALSE)
}

for (lv in c("G", "CT", "I")) {
  for (pos in c(TRUE, FALSE)) {
    nm <- paste0("ref_genes_", lv, "_", ifelse(pos, "pos", "neg"))
    res <- tryCatch(
      nicheDE::get_niche_DE_genes(obj, lv, index = index_ct, niche = niche_ct,
                                  positive = pos, alpha = 0.05),
      error = function(e) data.frame())
    wr_df(nm, res)
  }
}

mk <- tryCatch(
  nicheDE::niche_DE_markers(obj, index_ct, niche_ct, niche_ct2, alpha = 0.05),
  error = function(e) data.frame())
wr_df("ref_markers", mk)

# contrast_post raw p-values for every gene (kernel 1), for a direct parity probe
betas_all <- lapply(obj@niche_DE[[1]], function(r) r$betas)
vcov_all  <- lapply(obj@niche_DE[[1]], function(r) r$Varcov)
nulls_all <- lapply(obj@niche_DE[[1]], function(r) r$nulls)
ii <- which(obj@cell_types == index_ct)
n1 <- which(obj@cell_types == niche_ct)
n2 <- which(obj@cell_types == niche_ct2)
cp <- nicheDE::contrast_post(betas_all, vcov_all, nulls_all, ii, c(n1, n2))
dump_vec("ref_contrast_post", as.numeric(cp))
meta[["contrast_index"]]  <- ii
meta[["contrast_niche1"]] <- n1
meta[["contrast_niche2"]] <- n2

# check_colloc
cc <- nicheDE::check_colloc(obj, ii, n1)
dump_vec("ref_check_colloc", as.numeric(cc))

# --- Step 5: niche-LR ------------------------------------------------------
data("niche_net_ligand_target_matrix")
data("ramilowski_ligand_receptor_list")
lr_spot <- tryCatch(
  nicheDE::niche_LR_spot(obj, ligand_cell = niche_ct, receptor_cell = index_ct,
                         ligand_target_matrix = niche_net_ligand_target_matrix,
                         lr_mat = ramilowski_ligand_receptor_list,
                         K = 25, M = 50, alpha = 0.05, truncation_value = 3),
  error = function(e) { cat("niche_LR_spot error:", conditionMessage(e), "\n"); NULL })
if (!is.null(lr_spot)) wr_df("ref_niche_LR_spot", as.data.frame(lr_spot))

lr_cell <- tryCatch(
  nicheDE::niche_LR_cell(obj, ligand_cell = niche_ct, receptor_cell = index_ct,
                         ligand_target_matrix = niche_net_ligand_target_matrix,
                         lr_mat = ramilowski_ligand_receptor_list,
                         K = 25, M = 50, alpha = 0.05, alpha_2 = 0.5,
                         truncation_value = 3),
  error = function(e) { cat("niche_LR_cell error:", conditionMessage(e), "\n"); NULL })
if (!is.null(lr_cell)) wr_df("ref_niche_LR_cell", as.data.frame(lr_cell))

# --- Step 6: MergeObjects / Filter_NDE probes -------------------------------
o1 <- nicheDE::CreateNicheDEObject(counts, coord, libmat, deconv,
                                   sigma = sigma, Int = TRUE)
o2 <- nicheDE::CreateNicheDEObject(counts, coord * 2, libmat, deconv,
                                   sigma = sigma, Int = TRUE)
om <- nicheDE::MergeObjects(list(o1, o2))
dump_num("ref_merge_coord",     om@coord)
dump_num("ref_merge_num_cells", om@num_cells)
dump_vec("ref_merge_batch",     om@batch_ID)
meta[["merge_scale"]] <- om@scale

keep_cells <- obj@cell_names[seq(1, length(obj@cell_names), by = 3)]
of <- nicheDE::Filter_NDE(obj, keep_cells)
dump_num("ref_filter_num_cells", of@num_cells)
dump_num("ref_filter_en_1",      of@effective_niche[[1]])
meta[["filter_cells"]] <- keep_cells

# --- helper-function probes -------------------------------------------------
set.seed(42)
probe_T <- matrix(rnorm(49), 7, 7)
dump_num("probe_T", probe_T)
dump_num("ref_T_to_p_two",  nicheDE::T_to_p(probe_T, "two.sided"))
dump_num("ref_T_to_p_pos",  nicheDE::T_to_p(probe_T, "positive"))
dump_num("ref_T_to_p_neg",  nicheDE::T_to_p(probe_T, "negative"))
probe_M <- matrix(rnorm(49), 7, 7); probe_M[lower.tri(probe_M)] <- 0
dump_num("probe_M", probe_M)
dump_num("ref_ultosymmetric", nicheDE::ultosymmetric(probe_M))
probe_p <- runif(10); probe_w <- runif(10)
dump_vec("probe_p", probe_p); dump_vec("probe_w", probe_w)
dump_vec("ref_gene_level", nicheDE::gene_level(probe_p, probe_w))
probe_pm <- matrix(runif(50), 10, 5); probe_wc <- runif(5)
dump_num("probe_pm", probe_pm); dump_vec("probe_wc", probe_wc)
dump_vec("ref_celltype_level", nicheDE::celltype_level(probe_pm, probe_wc))
dump_vec("ref_nb_lik", nicheDE::nb_lik(c(1, 5, 3, 0, 7), c(2, 4, 3, 1, 6), 1.7))

# poolr probes (the port re-derives poolr, so lock its numbers down too)
set.seed(7)
A <- matrix(rnorm(36), 6, 6); Rp <- cor(A %*% t(A) + diag(6))
Rp <- cov2cor(A %*% t(A) + diag(6))
dump_num("probe_poolr_R", Rp)
dump_num("ref_mvnconv_m2lp_s1", poolr::mvnconv(Rp, side = 1, target = "m2lp", cov2cor = FALSE))
dump_num("ref_mvnconv_m2lp_s2", poolr::mvnconv(Rp, side = 2, target = "m2lp", cov2cor = FALSE))
pp <- c(0.01, 0.2, 0.5, 0.001, 0.7, 0.03)
dump_vec("probe_poolr_p", pp)
cv <- poolr::mvnconv(Rp, side = 1, target = "m2lp", cov2cor = FALSE)
dump_vec("ref_fisher_generalized", poolr::fisher(pp, side = 1, R = cv, adjust = "generalized")$p)
# the full poolr lookup table, for the clean-room verification test
data(mvnlookup, package = "poolr")
dump_num("ref_mvnlookup", as.matrix(mvnlookup))

# --- write meta ------------------------------------------------------------
write(jsonlite::toJSON(meta, auto_unbox = TRUE, digits = NA, null = "null"),
      file.path(outdir, "meta.json"))
cat("DONE ->", outdir, "\n")
