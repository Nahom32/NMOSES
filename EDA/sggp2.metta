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
(= (concatTuple $x $y) (collapse (union (superpose $x) (superpose $y))))
(= (initializeAlphabetSpace $alphabetSpace)
    (add-reduct $alphabetSpace (superpose (start expr bop uop term)))
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
(= (addTerminalsToSpace $terminals $space)
    (add-atom $space (term (superpose $terminals)))
)
(= (updateAtomSpace $space $previousRule $currentRule)
    (let () (remove-atom $space $previousRule)
        (add-atom $space $currentRule)
    )
)
(= (selectK $list $k $accum)
    (if ( <= (length $list) $k)
        $list
        (if (== $k 0)
            $accum
            (let* (
                ($head (car-atom $list)) 
                ($tail (cdr-atom $list)) 
            )
                (selectK $tail (- $k 1) (cons-atom $head $accum))
            )
        )

    )
)
(= (split-at-pointK $atoms $index $accum)
        (if (>= $index 0)
                (let* (
                        ($head (car-atom $atoms))
                        ($tail (cdr-atom $atoms))

                )
                        (split-at-pointK $tail (- $index 1) (cons-atom $head $accum))
                )
                ($accum $atoms)
        )
)
;; shuffles the the first k atoms with the (n-k) last list values having length n. 
(= (random-shuffleK $atoms)
        (let* (
                ($len (length $atoms))
                ($splitPoint (rand 0 (- $len 1)))
                (($fir $las) (split-at-pointK $atoms $splitPoint ()))
              )
                (collapse (union (superpose $las) (superpose $fir)))
        )
)
;; applies randomization to the list probabilistically
(= (randomizeList $atoms)
        (if (isEven (rand 0 10))
                (random-shuffleK $atoms)
                $atoms
        )
)
(= (isAtomJunctor $token)
    (if (or (== $token AND) (== $token OR)) True False)
)
;; checks for an availability of an even number.

(= (isEven $number)
        (if (== (% $number 2) 0)
                True 
                False
        )
)
(= (findSuitableProdRules $lhs $stocSpace)
    (collapse (match $stocSpace ($lhs $rhs $weight) ($lhs $rhs $weight)))
)


(= (getRuleFromSubTree $expression $grammarSpace)
    (unify $expression ($op $exp1 $exp2)
        (match $grammarSpace (, ($opType $op)
                                ($exp ($opType $exp $exp))

                        )
                        (($opType $op) ($exp ($opType $exp $exp)))

        )
        (unify $expression ($op $exp1)
            (match $grammarSpace (, 
                                ($opType $op) 
                                ($exp ($opType $exp))
                            )
                                (($exp ($opType $op)) ($exp ($opType $exp)))
                            )
            (match $grammarSpace ($parent $expression)
                ($parent $expression) 
            )
        )
         
    )

)
;; (: getRelations (-> Expression Expression Expression))
;; (= (getRelations $tree $accum)
;;     (if (== $tree ())
;;         $accum
;;         (let* (
;;             ($head (car-atom $tree))
;;             ($tail (cdr-atom $tree))
;;         )
;;             (if (isAtomJunctor $head)
;;                 (let* (
;;                     ($left (car-atom $tail))
;;                     ($secondTail (cdr-atom $tail))
;;                     ($right (car-atom $left))
;;                     ($newAccum (cons-atom ($head $left $right) $accum))
;;                 )
;;                     (getRelations $tail $newAccum)
;;                 )
;;                 (if (== $head NOT)
;;                     (let* ( 
;;                         ($child (car-atom $tail))
;;                         ($newAccum (cons-atom ($head $child) $accum))
;;                         )
;;                         (getRelations $tail $newAccum)
;;                     )
;;                     (let $newAccum (cons-atom $head $accum)
;;                         (getRelations $tail $newAccum)
;;                     )
;;                 )
;;             )
;;         )
;;     )
;; )
(= (findSuitableProdRules $lhs $stocSpace)
    (collapse (match $stocSpace ($lhs $rhs $weight) ($lhs $rhs $weight)))
)
(= (getRelations $tree $accum)
    (if (== $tree ())  
        $accum 
        (let* (
            ($head (car-atom $tree))
            ($tail (cdr-atom $tree))
        )
            (if (== (get-metatype $head) Expression)
                (let* (
                    ($recur-result (getRelations $head ()))
                    ($newAccum (concatTuple $accum $recur-result))
                )
                    (getRelations $tail $newAccum)
                )
                (if (isAtomJunctor $head)
                    (let* (
                        ($left (car-atom $tail))
                        ($tail' (cdr-atom $tail))
                        ($right (car-atom $tail'))
                        ($newAccum (cons-atom ($head $left $right) $accum))
                    )
                        (getRelations $tail $newAccum)
                    )
                    (if (== $head NOT)
                        (let* (
                            ($child (car-atom $tail))
                            ($newAccum (cons-atom ($head $child) $accum))

                        )
                            (getRelations $tail $newAccum)
                        )
                        (let $newAccum (cons-atom $head $accum)
                            (getRelations $tail $newAccum)
                        )
                    )

                )
            )
        )
    )
)
;; a helper that extracts the best after comparing with the threshold.
(= (filter $compSign $func $expression $threshold)
    (if ( $compSign ($func $expression) $threshold)
        (empty)
        $expression
    )
)
(= (filterList $compSign $func $expressions $threshold)
    (collapse (filter $compSign $func (superpose $expressions) $threshold))
)
;;This procedure implements the reinforcing the production rule that is available.
(= (reinforceProd ($lhs $rhs $weight) $learnRate $cardinality $stocSpace)
    (let ($parent $child $updatedWeight) (updateWeight ($lhs $rhs $weight) $learnRate $cardinality inc)
        (updateAtomSpace $stocSpace ($lhs $rhs $weight) ($parent $child $updatedWeight))
    )
)
;; This procedure implements the punishing a specific production rule that is available.
(= (punishProd ($lhs $rhs $weight) $learnRate $cardinality $stocSpace)
    (let ($parent $child $updatedWeight) (updateWeight ($lhs $rhs $weight) $learnRate $cardinality dec)
        (updateAtomSpace $stocSpace ($lhs $rhs $weight) ($parent $child $updatedWeight))
    );; (exploitInstances $rhs $grammarSpace $stocSpace $explorationCoeff (- $depth 1))
)
;; This procedure implements the reinforcement of the reinforcement of a grammar given a Tree
(= (reinforceGrammarbasedOnTree $tree $learnRate $cardinality $stocSpace)
    (let* (
        ($relations (generateRelations $tree NONE ()))
        ($prodRules (collapse (getprodRule (superpose $relations))))
        
    )
        (collapse (reinforceProd (superpose $prodRules) $learnRate $cardinality $stocSpace))
    )
)
;; The below piece of code punishes the production rule in a stochastic grammar based on a tree
(= (punishGrammarbasedOnTree $tree $learnRate $cardinality $stocSpace)
    (let* (
        ($relations (generateRelations $tree NONE ()))
        ($prodRules (collapse (getProdRule (superpose $relations))))
    )
        (collapse (punishProd (superpose $prodRules) $learnRate $cardinality $stocSpace))
    )
)


!(initializeAlphabetSpace &alphabetSpace)
!(initializeGrammarSpace &grammarSpace)
;; !(filterProductionByLeftHandSide &grammarSpace start)
;; !(filterProductionByLeftHandSide &grammarSpace expr)
;; !(addTerminalsToSpace (x y z a) &grammarSpace)
!(initializeStochasticSpace &alphabetSpace &grammarSpace &stochasticGrammarSpace)
!(getRuleFromSubTree (AND exp1 exp2) &grammarSpace)
!(getRelations (AND x1 (OR x2 x3)) ())

;; !(collapse (get-atoms &stochasticGrammarSpace))




