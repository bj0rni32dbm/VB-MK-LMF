include("utils.jl")
include("eval.jl")

sims1        = ["data/nr/mat/nr_simmat_dc_morgan_rbf.txt",
                "data/nr/mat/nr_simmat_dc_morgan_tanimoto.txt",
                "data/nr/mat/nr_simmat_dc_maccs_rbf.txt",
                "data/nr/mat/nr_simmat_dc_maccs_tanimoto.txt",
                "data/nr/mat/nr_simmat_dc.txt"]
sims2        = ["data/nr/mat/nr_simmat_dg.txt"]
interactions = "data/nr/mat/nr_admat_dgc.txt"
params       = Params(5, # c
                      1.0, # alpha_u
                      0.1, # alpha_v
                      1.0, # a_u
                      1.0, # a_v
                      100, # b_u
                      100, # b_v
                      20,  # iters
                      20)  # L

nsim1 = length(sims1)
nsim2 = length(sims2)
R_ref,Ku,Kv = importData(interactions,sims1,sims2)
Ku = [normalize_kernel(Ku[n],3) for n in 1:nsim1]
Kv = [normalize_kernel(Kv[n],3) for n in 1:nsim2]

function evaluate(test_idx,params)
	R=full(R_ref)
  for i in test_idx
    R[i[1],i[2]]=0.0
  end
  
  I,J = size(R)
  
  U=zeros(Float32,params.L,I)
  V=zeros(Float32,params.L,J)
  g_u=zeros(Float32,nsim1)
  g_v=zeros(Float32,nsim2)
  
  Ku_ptr=Array{Ptr{Float32},1}()
  Kv_ptr=Array{Ptr{Float32},1}()
  for i in 1:nsim1
  	push!(Ku_ptr,pointer(Ku[i]))
  end
  for i in 1:nsim2
  	push!(Kv_ptr,pointer(Kv[i]))
  end
  
  ccall((:factorize,"lib/libcuvbmf.so"),Void,(Int32,Int32,Ptr{Float32},Int32,Int32,Ptr{Ptr{Float32}},Ptr{Ptr{Float32}},Ptr{Float32},Ptr{Float32},Ptr{Float32},Ptr{Float32},Ptr{Params}),I,J,R,nsim1,nsim2,Ku_ptr,Kv_ptr,U,V,g_u,g_v,&params)
  
  ref=zeros(length(test_idx))
  res=zeros(length(test_idx))
  for i in 1:length(test_idx)
    ref[i]=R_ref[test_idx[i][1],test_idx[i][2]]
  end
  
  for i in 1:length(test_idx)
    mn=dot(U[:,test_idx[i][1]],V[:,test_idx[i][2]])
    res[i] = (exp(mn)/(1.0+exp(mn)))^(1.0)
  end

  res,ref
end

folds = importFolds("data/nr/cv/nr_all_folds_cvs1.txt")
num_folds = length(folds)
auprc_all = zeros(num_folds)
auroc_all = zeros(num_folds)

for f in 1:num_folds
    print("Fold ",f," ")
    res,ref = evaluate(folds[f],params)
    prec,rec,fpr,auprc,auroc=prc(res,ref)
    auprc_all[f]=auprc
    auroc_all[f]=auroc
    print("AUPRC: ",auprc,", AUROC: ",auroc,"\n")
end
print(mean(auprc_all),"\n")
print(mean(auroc_all),"\n")
