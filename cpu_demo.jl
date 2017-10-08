include("utils.jl")
include("fact.jl")
include("eval.jl")

similarities1 = [#"data/gpcr/mat/gpcr_simmat_dc_morgan_rbf.txt",
                 #"data/gpcr/mat/gpcr_simmat_dc_morgan_tanimoto.txt",
                 #"data/gpcr/mat/gpcr_simmat_dc_maccs_rbf.txt",
                 #"data/gpcr/mat/gpcr_simmat_dc_maccs_tanimoto.txt",
                 "data/gpcr/mat/gpcr_simmat_dc.txt"]
similarities2 = ["data/gpcr/mat/gpcr_simmat_dg.txt"]
interactions  = "data/gpcr/mat/gpcr_admat_dgc.txt"
folds         = importFolds("data/gpcr/cv/gpcr_all_folds_cvs1.txt")
parameters    = Params(10, # c
                      1.0, # alpha_u
                      0.1, # alpha_v
                      1.0, # a_u
                      1.0, # a_v
                      50,  # b_u
                      10,  # b_v
                      10,  # iters
                      20)  # L
nb_U          =       3    # num neighbors (kernels for U)
nb_V          =       3    # num neighbors (kernels for V)

reference,kernels_u,kernels_v = importData(interactions,similarities1,similarities2)
kernels_u = [normalize_kernel(K,nb_U) for K in kernels_u]
kernels_v = [normalize_kernel(K,nb_V) for K in kernels_v]
num_folds = length(folds)
auprc_all = zeros(num_folds)
auroc_all = zeros(num_folds)

for f in 1:num_folds
	test_idx = folds[f]
	R = full(reference)
	for i in test_idx
		R[i[1],i[2]] = 0.0
	end
	
	U,V,gamma_u,gamma_v = fct(R,parameters,kernels_u,kernels_v)
	
	ref  = zeros(length(test_idx))
  pred = zeros(length(test_idx))
  for i in 1:length(test_idx)
    ref[i] = reference[test_idx[i][1],test_idx[i][2]]
  end
  
  for i in 1:length(test_idx)
    mean    = dot(U[:,test_idx[i][1]],V[:,test_idx[i][2]])
    pred[i] = exp(mean)/(1.0+exp(mean))
  end
  
  prec,rec,fpr,auprc,auroc = prc(pred,ref)
  auroc_all[f] = auroc
  auprc_all[f] = auprc
  print("AUPRC: ",auprc,", AUROC: ",auroc,"\n")
end

println("-------")
println("\nAUPRC: ",mean(auprc_all)," ± ",2*std(auprc_all)/sqrt(num_folds))
println("\nAUROC: ",mean(auroc_all)," ± ",2*std(auroc_all)/sqrt(num_folds))
