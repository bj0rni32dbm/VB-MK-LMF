function rates(RR,R,limit)
	pr=RR.>=limit
	rl=R.>=0.5
	TP=sum( pr &  rl)
	FP=sum( pr & !rl)
	TN=sum(!pr & !rl)
	FN=sum(!pr &  rl)
	
	TP,FP,TN,FN
end

function precrec(RR,R,limit)
	TP,FP,TN,FN=rates(RR,R,limit)
	prec=TP/(TP+FP)
	rec=TP/(TP+FN)
	fpr=FP/(FP+TN)
	prec,rec,fpr
end

function prc(RR,R)
	limits=[minimum(RR):0.01:maximum(RR)...]
	precision=[0.0]
	recall=[1.0]
	false_positive_rate=[0.0]
	auprc=0.0
	auroc=0.0
	for l in 1:length(limits)
		prec,rec,fpr = precrec(RR,R,limits[l])
		push!(precision,prec)
		push!(recall,rec)
		push!(false_positive_rate,fpr)
		auprc+=(prec-precision[l])*(rec+recall[l])/2
		auroc+=(fpr-false_positive_rate[l])*(rec+recall[l])/2
	end
	precision[2:end],recall[2:end],false_positive_rate[2:end],auprc,(1.0-auroc)
end
