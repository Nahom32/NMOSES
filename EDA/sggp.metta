!(bind! &grammarSpace (new-space))
!(bind! &stochasticGrammarSpace (new-space))
!(bind! rand (py-atom random.randint)) ;;pseudo random integers
!(bind! random (py-atom random.random)) ;;pseudo random real numbers from 0 to 1
!(bind! pow (py-atom pow))
!(bind! &alphabetSpace (new-space))
;;concatenates two tuples.
 (= (concatTuple $x $y) (collapse (union (superpose $x) (superpose $y))))


;;calculates the length of a list
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
    (let $atomList (collapse (match $space $x $x))
        (length $atomList)
    )
)
;; the below code initializes the alphabet space containing the set {AND, OR, NOT, junctor}
;; the starting symbol will be junctor.
(= (initializeAlphabetSpace $space)
    (let () (add-atom $space junctor)
        (let () (add-atom $space AND)
            (let () (add-atom $space OR)
            ;;    (let () 
                    (add-atom $space NOT)
                    ;; (add-atom $space terminal)
            ;;    )
            )
        )
    )
)
;;This adds new grammar production rules into space
(= (initializeGrammarSpace $space)
    (let () (add-atom $space (junctor junctor))
        (let () (add-atom $space (junctor AND))
            (let () (add-atom $space (junctor OR))
                (let () (add-atom $space (AND terminal))
                    (let () (add-atom $space (OR terminal))
                        (add-atom $space (NOT terminal))
                    )
                )
            )

        )
    )

)
;;The below function adds list production items to space
(= (addListToSpace $space $list)
    (if (== $list ())
        (empty)
        (let* (
            ($head (car-atom $list))
            ($tail (cdr-atom $list))
            (() (add-atom $space $head))
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
    (if (== $tuple ())
        0
        (+ (car-atom $tuple) (sum (cdr-atom $tuple)))
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
            (let $updatedWeight (/ $weight (pow (- 1 $learnRate) $cardinality));;an update mechanism if production rule isn't chosen
                ($lhs $rhs $updatedWeight)
            )
        )
    )
)
;;This function is used to extract a parent or left hand side of a context free grammar.
(= (extractParent $exp)
    (let ($lhs $rhs) $exp
        $lhs
    )
)
;; the below function initializes a stochastic space.
(= (initializeStochasticSpace $alphabetSpace $grammarSpace $stocSpace)
    (let* (
        ($grammarAtoms (extractAtomSpace $alphabetSpace))
        ($weightedAtoms (collapse (initializeForSingleLhs $grammarSpace (superpose $grammarAtoms))))
    )
        (addListToSpace $stocSpace (superpose $weightedAtoms))
    )
    
)


;; mutateRule mutates a production rule with some mutation rate.
(= (mutateRule $prodRule $mutationRate)
    (let* (
        (($lhs $rhs $weight) $prodRule) 
        ($updatedWeight (* $weight $mutationRate))
        )
        ($lhs $rhs $updatedWeight)
        
    )
)
;;This function extracts the weights of production rules from the atomspace 
;;The parent represents a left hand side production rule
(= (extractWeightsFromParent $parent $space)
    (collapse (match $space ($parent $child $value) $value))
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
;; returns a non-deterministic instances stochastically
(= (generateInstancesHelper $initSymbol $stocSpace $depth)
    (if (>= $depth 0)
        (if (== $initSymbol terminal)
            $initSymbol
            (match $stocSpace ($initSymbol $rhs $value)
                (if (== $rhs junctor)
                    (generateInstancesHelper $rhs $stocSpace $depth)
                    (let* (
                        ($prodProb (calculateProbability ($initSymbol $rhs) $stocSpace))
                        ($randomNumber (random))
                        )
                        (if (>= $prodProb $randomNumber)
                                ($rhs (generateInstancesHelper $rhs $stocSpace (- $depth 1)) (generateInstancesHelper $rhs $stocSpace (- $depth 1)))
                                (generateInstancesHelper $rhs $stocSpace (- $depth 1))
                        )
                    )
                )
            )
        )
        (empty)
    )
)
;; checks whether an atom is a type of junctor
(=(isAtomJunctor $node)
    (if (or (== $node AND) (== $node OR))
        True
        False
    )
)
(= (generateInstances $initSymbol $space $depth)
    (collapse (generateInstancesHelper $initSymbol $space $depth))
)
(= (getProdRule ($parent $child) $space)
    (let* ( 
        ($isParentJunctor (or (== $parent AND) (== $parent OR)))
        ($isChildJunctor (or (== $child AND) (== $child OR)))
    )
    (if (and $isParentJunctor $isChildJunctor)
        (match $space (junctor junctor $value) (junctor junctor $value))
        (if (and $isParentJunctor (not $isChildJunctor))
            (match $space ($parent terminal $value) ($parent terminal $value))
            (empty)
        )
    )
    
    )

)
;; The below implementation generates a set of relations of parents and childs
(= (generateRelations $tree $parent $accum)
    (if (== $tree ())
        $accum
        (let* (
            ($head (car-atom $tree))
            ($tail (cdr-atom $tree))
            ($typeofHead (get-metatype $head))
        )
            (if (== $parent NONE)
                (generateRelations $tail $head $accum)
                (if (== $typeofHead Expression)
                    (let* (
                        ($resultofSubTree (generateRelations $head $parent ()))
                        ($newAccum (concatTuple $accum $resultofSubTree))
                    )
                        (generateRelations $tail $parent $newAccum)
                    )
                    (let $newAccum (cons-atom ($parent $head) $accum)
                        (if (isAtomJunctor $head)
                            (generateRelations $tail $head $newAccum) ;;change of parent since we found a new junctor
                            (generateRelations $tail $parent $newAccum) ;;parent still remains since we haven't found a junctor.
                        )
                    )
                    
                )
            )
        )
    )
)



;; !(sum (1 2 2 4 5))
!(initializeAlphabetSpace &alphabetSpace)
!(extractAtomSpace &alphabetSpace)
!(initializeGrammarSpace &grammarSpace) 
;;!(filterProductionByLeftHandSide &grammarSpace terminal)
!(updateWeight (AND terminal 6) 0.001 7 dec)
!(mutateRule (AND terminal 4) 0.7)
!(initializeStochasticSpace &alphabetSpace &grammarSpace &stochasticGrammarSpace)
!(extractAtomSpace &stochasticGrammarSpace)
!(calculateProbability (junctor OR) &stochasticGrammarSpace)
!(getProdRule (AND x) &stochasticGrammarSpace)
!(generateRelations (AND (OR x y) x) NONE ()) 

;;!(generateInstances junctor &stochasticGrammarSpace 3)
;; !(initializeStochasticSpace &grammarSpace)

;; (= (calculateProbability $space $productionRule)
;;     ()
;; )

;; (= (initializeSGSpace $grammarSpace $sgSpace)
;;    () 
;; )
;; !(rand 5 15)

