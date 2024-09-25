(= (terminalSet) A)
(= (terminalSet) B)
(= (terminalSet) X)
(= (terminalSet) Y)


(= (functionSet) AND)
(= (functionSet) OR)
(= (functionSet) NOT)


(= (booleanArity NOT) 1)
(= (booleanArity OR)  2)
(= (booleanArity AND) 2)

(= (chooseRandom $x)
    (let $collapsed (collapse ($x))
       ((py-atom random.choice) (py-list $collapsed))
    )
)

(= (generateFullTree $depth $funcSet $termSet)
	 (if (== $depth 0)
		 (chooseRandom $termSet)
		 (OR 
			(generateFullTree (- $depth 1) $funcSet $termSet)
			(generateFullTree (- $depth 1) $funcSet $termSet)
		 )
	 )
)


;!(terminalSet)
;!(terminalSet)
!(generateFullTree 2 functionSet terminalSet)
