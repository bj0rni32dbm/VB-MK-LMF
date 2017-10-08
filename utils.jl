function importData(interactions, sim1, sim2)
	rows = Int32[]
	cols = Int32[]
	vals = Float32[]
	
	f = open(interactions)
	lines  = readlines(f)
	rnames = split(chomp(lines[1]),'\t')[2:end]
	rnames = Dict{String,Int32}(zip(rnames,1:length(rnames)))
	cnames = Dict{String,Int32}()
	for i in 2:length(lines)
		recs = split(chomp(lines[i]),'\t')
		cnames[recs[1]]=i-1
		for j in 2:length(recs)
			push!(rows,j-1)
			push!(cols,i-1)
			push!(vals,parse(Float32,recs[j]))
		end
	end
	close(f)
	
	R   = sparse(rows,cols,vals)
	I,J = size(R)
	
	Ku = [zeros(Float32,I,I) for i in 1:length(sim1)]
	for k in 1:length(sim1)
		f      = open(sim1[k])
		lines  = readlines(f)
		knames = split(chomp(lines[1]),'\t')[2:end]
		
		for i in 2:length(lines)
			recs = split(chomp(lines[i]),'\t')
			id1  = rnames[recs[1]]
			for j in 2:length(recs)
				Ku[k][id1,rnames[knames[j-1]]] = parse(Float32,recs[j])
			end
		end
		close(f)
	end
	
	Kv = [zeros(Float32,J,J) for i in 1:length(sim2)]
	for k in 1:length(sim2)
		f      = open(sim2[k])
		lines  = readlines(f)
		knames = split(chomp(lines[1]),'\t')[2:end]
		
		for i in 2:length(lines)
			recs = split(chomp(lines[i]),'\t')
			id1  = cnames[recs[1]]
			for j in 2:length(recs)
				Kv[k][id1,cnames[knames[j-1]]] = parse(Float32,recs[j])
			end
		end
		close(f)
	end
	
	R,Ku,Kv
end

function importFolds(path)
	f     = open(path)
	lines = readlines(f)
	folds = Array{Array{Tuple{Int32,Int32},1},1}()
	for line in lines
		rec  = split(chomp(line),",")
		curr = Array{Tuple{Int32,Int32},1}()
		for r in 1:(Int32(length(rec)/2))
			push!(curr,(parse(Int32,rec[(r-1)*2+1]),parse(Int32,rec[r*2])))
		end
		push!(folds,curr)
	end
	close(f)
	folds
end

function normalize_kernel(K,num_neighbors)
	s  = size(K,1)
	K2 = deepcopy(K)
	for l in 1:s
		ind = sortperm(-K2[l,:])[(num_neighbors+1):end]
		K2[l,ind]=0.01*K2[l,ind]
		K2[ind,l]=0.01*K2[ind,l]
	end
	K2/trace(K2)
end

type Params
	c::Float32
	alpha_u::Float32
	alpha_v::Float32
	a_u::Float32
	a_v::Float32
	b_u::Float32
	b_v::Float32
	iters::Int
	L::Int
end
