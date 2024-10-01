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

(= (length $x)
    (if (== () $x)
        0
        (+ 1 (length (cdr-atom $x)))
    )
)

(= (getByIndex $collapsedVal $n) 
    (if (== $n 0)
        (car-atom $collapsedVal)
        (getByIndex (cdr-atom $collapsedVal) (- $n 1))
    )
)

(= (chooseRandom $x)
    (let $collapsed (collapse ($x))
       (getByIndex $collapsed ((py-atom random.randint) 0 (- (length $collapsed) 1)))
    )
)

(= (generateFullTree $depth $funcSet $termSet)
	 (if (== $depth 0)
		 (chooseRandom $termSet)
		 (let $func (chooseRandom $funcSet)
			 (if (== (booleanArity $func) 1)
				( $func (generateFullTree (- $depth 1) $funcSet $termSet))
				( $func (generateFullTree (- $depth 1) $funcSet $termSet) (generateFullTree (- $depth 1) $funcSet $termSet) )
			 )
		 )
	 )
)


;!(terminalSet)
;!(terminalSet)
;!(length (A B C)) ; should be 3
;!(getByIndex (A B C) 2) ; should be C
!(generateFullTree 2 functionSet terminalSet)
