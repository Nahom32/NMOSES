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

(= (generateGrowTree $depth $funcSet $termSet)
	 (if (or (== $depth 0) (< ((py-atom random.random)) 0.5))
		 (chooseRandom $termSet)
		 (let $func (chooseRandom $funcSet)
			 (if (== (booleanArity $func) 1)
				( $func (generateGrowTree (- $depth 1) $funcSet $termSet))
				( $func (generateGrowTree (- $depth 1) $funcSet $termSet) (generateGrowTree (- $depth 1) $funcSet $termSet) )
			 )
		 )
	 )
)

(= (rampedHalfHalf  $popSize $depth $population $funcSet $termSet $generateFullTree $generateGrowTree )
	 (if (== $depth 0)
		 ()
		 (let $generate (rp ((py-atom math.floor) (/ $popSize (* $depth 2))) $depth $population $funcSet $termSet $generateFullTree $generateGrowTree)
            (rampedHalfHalf $popSize (- $depth 1) $population $funcSet $termSet $generateFullTree $generateGrowTree)
		 )
	 )
)

(= (rp $range $depth $population $funcSet $termSet $generateFullTree $generateGrowTree )
	 (if (<= $range 0)
		 ()
		 (let*
			 (
				 ($full (add-atom &self (= ($population) ($generateFullTree $depth $funcSet $termSet))))
				 ($grow (add-atom &self (= ($population) ($generateGrowTree $depth $funcSet $termSet))))
			 )
			 (
				rp (- $range 1) $depth $population $funcSet $termSet $generateFullTree $generateGrowTree
			 )
		 )
	 )
) 




;!(terminalSet)
	;!(terminalSet)
	;!(length (A B C)) ; should be 3
;!(getByIndex (A B C) 2) ; should be C
;!(generateFullTree 2 functionSet terminalSet)
;!(generateGrowTree 2 functionSet terminalSet)
(= (population) ())
;!(rp 2 population functionSet terminalSet generateFullTree generateGrowTree)
!(rampedHalfHalf 4 8 population functionSet terminalSet generateFullTree generateGrowTree)
!(population)


