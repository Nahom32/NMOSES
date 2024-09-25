(= (terminalSet) (superpose (A B C)))
(= (functionSet) (superpose (AND OR NOT)))


(= (booleanArity NOT) 1)
(= (booleanArity OR)  2)
(= (booleanArity AND) 2)


(= (generateFullTree $depth $funcSet $termSet)
	 (if (== $depth 0)
		 (A)
		 (OR 
			(generateFullTree (- $depth 1) $funcSet $termSet)
			(generateFullTree (- $depth 1) $funcSet $termSet)
		 )
	 )
)


!(booleanArity AND)
!(booleanArity OR)
!(booleanArity NOT)

!(generateFullTree 5 $functionSet $terminalSet)
