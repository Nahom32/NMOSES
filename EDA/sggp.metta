!(bind! &grammarSpace (new-space))
!(bind! &stochasticGrammarSpace (new-space))
!(bind! rand (py-atom random.randint))
!(bind! pow (py-atom pow))
(= (length $list)
    (if (== $list ())
        0
        (+ 1 (length (cdr-atom $list)))
    )
)
;;the below function extracts the atoms from an atomspace.
(=(extractAtomSpace $space)
    (collapse (match $space $value $value))
)
;; the below function calculates the number of atoms in an atomspace
(= (lengthAtomSpace $space) 
    (let $atomList (collapse (match $space $atomGrammar $atomGrammar))
        (length $atomList)
    )
)
;;This adds new grammar into space
(= (initializeGrammarSpace $space)
    (let () (add-atom $space (junctor junctor))
        (let () (add-atom $space (junctor AND))
            (let () (add-atom $space (junctor OR))
                (let () (add-atom $space (AND terminal))
                    (add-atom $space (OR terminal))
                )
            )

        )
    )

)
;;The below function adds list elements to space
(=(addListToSpace $space $list)
    (if (== $terminalSet ())
        ()
        (let* (
            ($head (car-atom $terminalSet))
            ($tail (cdr-atom $terminalSet))
            (() (add-atom $space (terminal $head)))
        )
            (addListToSpace $space $tail)
        )
    )
)
;;The below function filters a production rule by the left hand side.
(= (filterProductionByLeftHandSide $space $left)
    (collapse (match $space ($left $rhs) ($left $rhs)))
)
;; the below function assigns weight for a production rule.
(= (assignWeight $rule $weight)
    (let ($parent $child) $rule
        ($parent $child $weight)
    )
)
;;The below function initializes a weighted stochastic grammar for a single parent.
(= (initializeForSingleLhs $grammarSpace $parent)
    (let* (
            ($children (filterProductionByLeftHandSide $grammarSpace $parent))
            ($weight (rand 5 10)) ;;a random initial value to be assigned as weights
            ($weightedChildren (collapse (assignWeight (superpose $children) $weight)))
        )
        $weightedChildren
    )
)
;; the below function finds the sum of a given tuple.
(= (sum $tuple)
    (let* (
        ($head (car-atom $tuple))
        ($tail (cdr-atom $tuple))
    )
        (if (== $tuple ())
            0
            (+ $sum (add $tail))
        )
    )
)
;;The updateWeight function is how the guided search is modified
;;The parameters are the production rule
;;a learn rate on how the weight changes 0 < lr < 1
;;cardinality is the number of chosen weights according to a probabilistic value.
;;type shows if it should be increased or decreased;
(= (updateWeight $prodRule $learnRate $cardinality $type)
    (let ($lhs $rhs $weight) $prodRule
        (if (== $type inc)
            (let $updatedWeight (* $weight (pow (+ 1 $learnRate) $cardinality));;an update rule if production rule is chosen
                ($lhs $rhs $updatedWeight)
            )
            (let $updatedWeight (/ $weight (pow (+ 1 $learnRate) $cardinality));;an update mechanism if production rule isn't chosen
                ($lhs $rhs $updatedWeight)
            )
        )
    )
)
(=(extractParent $exp)
    (let ($lhs $rhs) $exp
        $lhs
    )
)
(= (initializeStochasticSpace $grammarSpace)
    (let* (
        ($grammarAtoms (extractAtomSpace $grammarSpace))
        ($weightedAtoms (collapse (initializeForSingleLhs $grammarSpace (extractParent (superpose $grammarAtoms)))))
    )
        $weightedAtoms
    )
    
)
!(initializeGrammarSpace &grammarSpace)
!(initializeForSingleLhs &grammarSpace junctor)
!(updateWeight (AND terminal 6) 0.001 7 dec)
!(initializeStochasticSpace &grammarSpace)

;; (= (calculateProbability $space $productionRule)
;;     ()
;; )

;; (= (initializeSGSpace $grammarSpace $sgSpace)
;;    () 
;; )
;; !(rand 5 15)

