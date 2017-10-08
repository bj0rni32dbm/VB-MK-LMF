function fct(R,params,K_u,K_v)
	I,J   = size(R)
	U     = zeros(Float32,params.L,I)
	V     = zeros(Float32,params.L,J)
	nsim1 = length(K_u)
	nsim2 = length(K_v)
	g_u   = ones(Float32,nsim1)*(params.a_u/params.b_u)
	g_v   = ones(Float32,nsim2)*(params.a_v/params.b_v)

	m_u   = (sum(R,2).>0)*Float32(1.0)
	m_v   = (sum(R,1).>0)*Float32(1.0)

	for i in 1:I
		U[:,i] = randn(params.L)*params.alpha_u
	end

	for j in 1:J
		V[:,j] = randn(params.L)*params.alpha_v
	end

	R_hat = (1.0-params.c)*R-1.0
	mumv = m_u*m_v
	mcRm = (params.c*R + R_hat/2.0).*mumv
	mRm  = R_hat.*mumv

	Lap_u = [ diagm(vec(sum(K,1))) - K for K in K_u ]
	Lap_v = [ diagm(vec(sum(K,1))) - K for K in K_v ]

	L_u = zeros(Float32,params.L*I,params.L*I)
	L_v = zeros(Float32,params.L*J,params.L*J)
	BLAS.syr!('U',1.0f0,vec(U),L_u)
	BLAS.syr!('U',1.0f0,vec(V),L_v)

	for iter in 1:params.iters
		EuuT = reshape(diag(L_u),params.L,I)
		EvvT = reshape(diag(L_v),params.L,J)
		Xi = zeros(Float32,I,J)
		for i in 1:I
			for j in 1:J
				s = 0.0f0
				for l in 1:params.L
					var_u = EuuT[l,i] - U[l,i]*U[l,i]
					var_v = EvvT[l,j] - V[l,j]*V[l,j]
					s += (U[l,i]*V[l,j])^2 + U[l,i]^2*var_u + V[l,j]^2*var_v + var_u*var_v
				end
				Xi[i,j] = sqrt(s)
			end
		end
		mXRm = -(0.5./Xi.*(1.0./(1.0+exp(-Xi))-0.5)).*mRm
		gLaI_u = -0.5*(sum([Lap_u[n]*g_u[n] for n in 1:nsim1]) + params.alpha_u*eye(Float32,I))
		U = BLAS.gemm('N','T',1.0f0,V,mcRm)
		vU = vec(U)
		L_u = kron(-2.0*gLaI_u,eye(Float32,params.L))
		for i in 1:I
			s = zeros(params.L,params.L)
			for j in 1:J
				s += 2.0*mXRm[i,j]*L_v[((j-1)*params.L+1):(j*params.L),((j-1)*params.L+1):(j*params.L)]
			end
			L_u[((i-1)*params.L+1):(i*params.L),((i-1)*params.L+1):(i*params.L)] += s
		end
		LAPACK.potrf!('U',L_u)
		LAPACK.potrs!('U',L_u,vU)
		LAPACK.potri!('U',L_u)
		BLAS.syr!('U',1.0f0,vU,L_u)
		U = reshape(vU,params.L,I)
		L_u = full(Symmetric(L_u))

		EuuT = reshape(diag(L_u),params.L,I)
		EvvT = reshape(diag(L_v),params.L,J)
		Xi = zeros(Float32,I,J)
		for i in 1:I
			for j in 1:J
				s = 0.0f0
				for l in 1:params.L
					var_u = EuuT[l,i] - U[l,i]*U[l,i]
					var_v = EvvT[l,j] - V[l,j]*V[l,j]
					s += (U[l,i]*V[l,j])^2 + U[l,i]^2*var_u + V[l,j]^2*var_v + var_u*var_v
				end
				Xi[i,j] = sqrt(s)
			end
		end
		mXRm = -(0.5./Xi.*(1.0./(1.0+exp(-Xi))-0.5)).*mRm
		gLaI_v = -0.5*(sum([Lap_v[n]*g_v[n] for n in 1:nsim2]) + params.alpha_v*eye(Float32,J))
		V = BLAS.gemm('N','N',1.0f0,U,mcRm)
		vV = vec(V)
		L_v = kron(-2.0*gLaI_v,eye(Float32,params.L))
		for j in 1:J
			s = zeros(params.L,params.L)
			for i in 1:I
				s += 2.0*mXRm[i,j]*L_u[((i-1)*params.L+1):(i*params.L),((i-1)*params.L+1):(i*params.L)]
			end
			L_v[((j-1)*params.L+1):(j*params.L),((j-1)*params.L+1):(j*params.L)] += s
		end
		LAPACK.potrf!('U',L_v)
		LAPACK.potrs!('U',L_v,vV)
		LAPACK.potri!('U',L_v)
		BLAS.syr!('U',1.0f0,vV,L_v)
		V = reshape(vV,params.L,J)
		L_v = full(Symmetric(L_v))

		sb_u = zeros(Float32,I,I)
		for i in 1:I
			for j in (i+1):I
				sb_u[i,j] = sum(L_u[((i-1)*params.L+1):(i*params.L),((j-1)*params.L+1):(j*params.L)])
			end
			sb_u[i,i] = sum(L_u[((i-1)*params.L+1):(i*params.L),((i-1)*params.L+1):(i*params.L)])
		end
		sb_u  = full(Symmetric(sb_u))
		dsb_u = diag(sb_u)
		sb_u  = dsb_u*ones(Float32,I)'+ones(Float32,I)*dsb_u' - 2.0*sb_u
		g_u   = (params.a_u+I*I/2)./(params.b_u+[sum(K.*sb_u)/2 for K in K_u])

		sb_v = zeros(Float32,J,J)
		for i in 1:J
			for j in (i+1):J
				sb_v[i,j] = sum(L_v[((i-1)*params.L+1):(i*params.L),((j-1)*params.L+1):(j*params.L)])
			end
			sb_v[i,i] = sum(L_v[((i-1)*params.L+1):(i*params.L),((i-1)*params.L+1):(i*params.L)])
		end
		sb_v  = full(Symmetric(sb_v))
		dsb_v = diag(sb_v)
		sb_v  = dsb_v*ones(Float32,J)'+ones(Float32,J)*dsb_v' - 2.0*sb_v
		g_v   = (params.a_v+J*J/2)./(params.b_v+[sum(K.*sb_v)/2 for K in K_v])
	end
	U,V,g_u,g_v
end
