;; <start> -> <expr>
;; <expr>  -> <bop> <expr> <expr> |
;;            <uop> <expr> |
;;            <term>
;; <bop>   -> AND | OR
;; <uop>   -> NOT
!(bind! &grammarSpace (new-space))
!(bind! &stochasticGrammarSpace (new-space))
!(bind! &alphabetSpace (new-space))
!(bind! rand (py-atom random.randint)) ;;pseudo random integers
!(bind! random (py-atom random.random)) ;;pseudo random real numbers from 0 to 1
!(bind! pow (py-atom pow))
(= (initializeAlphabetSpace $alphabetSpace)
    (add-reduct $alphabetSpace (superpose (start expr bop uop)))
)

(= (initializeGrammarSpace $grammarSpace)
    (add-reduct $grammarSpace (superpose (
            (start expr)
            (expr (bop expr expr))
            (expr (uop expr))
            (expr term)
            (bop AND)
            (bop OR)
            (uop NOT)

        )
    )
)
)

(= (addListToSpace $space $list)
    (add-reduct $space (superpose $list))
)


(= (filterProductionByLeftHandSide $space $left)
    (collapse (match $space ($left $rhs) ($left $rhs)))
)
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
(= (updateWeight ($lhs $rhs $weight) $learnRate $cardinality inc)
    (let $updatedWeight (* $weight (pow (+ 1 $learnRate) $cardinality)) ($lhs $rhs $updatedWeight))
)
(= (updateWeight ($lhs $rhs $weight) $learnRate $cardinality dec)
    (let $updatedWeight (/ $weight (pow (- 1 $learnRate) $cardinality)) ($lhs $rhs $updatedWeight))
)
;;This function is used to extract a parent or left hand side of a context free grammar.
(= (extractParent ($lhs $rhs)) $lhs)
(= (extractParent ($lhs $rhs $weight)) $lhs)
;; the below function initializes a stochastic space.
(= (initializeStochasticSpace $alphabetSpace $grammarSpace $stocSpace)
    (let* (
        ($grammarAtoms (collapse (get-atoms $alphabetSpace)))
        ($weightedAtoms (collapse (initializeForSingleLhs $grammarSpace (superpose $grammarAtoms))))
    )
        (addListToSpace $stocSpace (superpose $weightedAtoms))
    )
    
)


;; mutateRule mutates a production rule with some mutation rate.
(= (mutateRule ($lhs $rhs $weight) $mutationRate) ($lhs $rhs (* $weight $mutationRate)))
;;This function extracts the weights of production rules from the atomspace 
;;The parent represents a left hand side production rule
(= (extractWeightsFromParent $parent $space)
    (collapse (match $space ($parent $child $weight) $weight))
)
;; This function calculates probability of a production rule
(= (calculateProbability ($lhs $rhs) $space)
    (let* (
        ($totalValue (sum (extractWeightsFromParent $lhs $space)))
        ($weight (match $space ($lhs $rhs $value) $value))
    ) 
        (/ $weight $totalValue)
    )
)
!(initializeAlphabetSpace &alphabetSpace)
!(initializeGrammarSpace &grammarSpace)
!(filterProductionByLeftHandSide &grammarSpace start)
!(filterProductionByLeftHandSide &grammarSpace expr)
!(initializeStochasticSpace &alphabetSpace &grammarSpace &stochasticGrammarSpace)
!(collapse (get-atoms &stochasticGrammarSpace))
;; !(collapse (get-atoms &grammarSpace))



