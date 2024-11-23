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
              (let () (add-atom $space NOT)
                    (add-atom $space terminal)
                )
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
;; (= (updateWeight $prodRule $learnRate $cardinality $type)
;;     (let ($lhs $rhs $weight) $prodRule
;;         (if (== $type inc)
;;             (let $updatedWeight (* $weight (pow (+ 1 $learnRate) $cardinality));;an update rule if production rule is chosen
;;                 ($lhs $rhs $updatedWeight)
;;             )
;;             (let $updatedWeight (/ $weight (pow (- 1 $learnRate) $cardinality));;an update mechanism if production rule isn't chosen
;;                 ($lhs $rhs $updatedWeight)
;;             )
;;         )
;;     )
;; )
;;The updateWeight function is how the guided search is modified
;;The parameters are the production rule
;;a learn rate on how the weight changes 0 < lr < 1
;;cardinality is the number of chosen weights according to a probabilistic value.
;;type shows if it should be increased or decreased;
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
        ($grammarAtoms (extractAtomSpace $alphabetSpace))
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
;; returns a non-deterministic instances stochastically
;; (= (generateInstancesHelper $initSymbol $stocSpace $depth)
;;     (if (> $depth 0)
;;         (if (== $initSymbol terminal)
;;             (match $stocSpace ($initSymbol $rhs $value) $rhs)
;;             (match $stocSpace ($initSymbol $rhs $value)
;;                  (if (== $rhs junctor)
;;                      (generateInstancesHelper $rhs $stocSpace (- $depth 1))
;;                     (let* (
;;                             ($prodProb (calculateProbability ($initSymbol $rhs) $stocSpace))
;;                             ($randomNumber (random))
;;                             ($isRhsJunctor (or (== $rhs OR) (== $rhs AND))) ;; check if the rhs is a junctor.
;;                         )
;;                         (if (and (>= $prodProb $randomNumber) $isRhsJunctor)
;;                                 ($rhs (generateInstancesHelper $rhs $stocSpace (- $depth 1)) (generateInstancesHelper $rhs $stocSpace (- $depth 1)))
;;                                 (generateInstancesHelper $rhs $stocSpace (- $depth 1))
;;                         )
;;                     )
;;             )
;;             )
;;         )
;;         (empty)
;;     )
;; )
;; =========================================================
;; algorithm for generating instances // Exploitation purposes
;; step 1 start from an init symbol with a depth = n.
;; match the grammar atomspace to get matching tokens as a right handside from the context free grammar
;; select probabilistically the production rule that passes the threshold
;; build the tree recursively and non-deterministically. with depth (n-1)
;; and make the rhs to lhs and start step 2
;; Does this procedure result in a well formed tree?
;; Do this until the depth is n = 0
;; after finishing collapse the non-deterministic result.
;; Question, Does this result in a well formed tree?
;; Definition: A well formed tree is a recursive data structure were internal nodes have atleast two children 
;; This stands out if the internal node is not a NOT junctor.
;; If the internal node is a NOT junctor it will have a one child

(= (generateInstancesHelper $initSymbol $grammarSpace $stocSpace $depth)
    (if (> $depth 0)
        (let $admissibleLhs (collapse (filterProductionByLeftHandSide $grammarSpace $initSymbol)) ;;check if there are any set of lhs elements
            (if (== $admissibleLhs ())
                $initSymbol
                (match $stocSpace ($initSymbol $rhs $weight)
                    (if (== $rhs junctor)
                        (if (> 0.5 (random)) ;; This logic is to be customized but designed to work temporarily
                            (AND (generateInstancesHelper $rhs $grammarSpace $stocSpace (- $depth 1)) (generateInstancesHelper $rhs $grammarSpace $stocSpace (- $depth 1)))
                            (OR (generateInstancesHelper $rhs $grammarSpace $stocSpace (- $depth 1)) (generateInstancesHelper $rhs $grammarSpace $stocSpace (- $depth 1))) 
                        )
                        ;; (let $probValue (calculateProbability ()))
                        (let () (println! (the calculatedProbability is (calculateProbability ($initSymbol $rhs) $stocSpace)))
                            (let* (
                                ($result (calculateProbability ($initSymbol $rhs) $stocSpace))
                                ($randomValue (random))
                                )
                                (if (and (>= $result $randomValue) (isAtomJunctor $rhs))
                                    ($rhs (generateInstancesHelper $rhs $grammarSpace $stocSpace (- $depth 1)) (generateInstancesHelper $rhs $grammarSpace $stocSpace (- $depth 1)))
                                    (generateInstancesHelper $rhs $grammarSpace $stocSpace (- $depth 1))
                                )
                            )
                        )
                    )
                    
                )
                

            )
        )
        (match $stocSpace (terminal $rhs $weight)
            (if (> (calculateProbability (terminal $rhs) $stocSpace) (random))
                $rhs
                (empty)
            )
        )
    )
)
(= (generateInstances $initSymbol $grammarSpace $stocSpace $depth)
    (collapse (generateInstancesHelper $initSymbol $grammarSpace $stocSpace $depth))
)
;; checks whether an atom is a type of junctor
(= (isAtomJunctor $node)
    (if (or (== $node AND) (== $node OR))
        True
        False
    )
)
;;gets a production rule from a (parent child) relationship in a syntax tree.
;; (AND (OR x y) z)
;; ((AND OR) (OR x) (OR y) (AND z))
;; (AND OR) (AND x) (OR)
(= (getProdRule ($parent $child) $space)
    (let* ( 
        ($isParentJunctor (or (== $parent AND) (== $parent OR)))
        ($isChildJunctor (or (== $child AND) (== $child OR)))
    )
    (if (and $isParentJunctor $isChildJunctor)
        (match $space (junctor $child $weight) (junctor $child $weight))
        (if (and $isParentJunctor (not $isChildJunctor))
            (match $space ($parent terminal $weight) ($parent terminal $weight))
            (empty)
        )
    )
    
    )

)
;; adds elements of terminal set to space to pertain to the grammar.
;; if terminals == (x y z a) (terminal x) (terminal y) (terminal z) and (terminal a) is going to be
;; persisted.
(= (addTerminalsToSpace $terminals $space)
    (add-atom $space (terminal (superpose $terminals)))
)
;; The below implementation generates a set of relations of parents and childs as a tuple or
;; an expression.
(: generateRelations (-> Expression Atom Expression Expression))
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
;; The below function updates the production rule given the previously available rule
;; and the current rule.
(= (updateAtomSpace $space $previousRule $currentRule)
    (let () (remove-atom $space $previousRule)
        (add-atom $space $currentRule)
    )
)

(= (isWellFormed $tree)
    (if (> (length $tree) 0)
        (let* (
                ($head (car-atom $tree))
                ($tail (cdr-atom $tree))
                ($isHeadJunctor (isAtomJunctor $head))
                ($lengthTail (length $tail))
            )
                (if (and $isHeadJunctor (>= (length $tail) 2))
                    (isWellFormed $tail)
                    (if $isHeadJunctor
                        False
                        True
                    )
                    
                )

        )
        True
    )
      
)




;; !(sum (1 2 2 4 5))
!(initializeAlphabetSpace &alphabetSpace)
!(extractAtomSpace &alphabetSpace)
!(initializeGrammarSpace &grammarSpace) 
!(filterProductionByLeftHandSide &grammarSpace terminal)
!(updateWeight (AND terminal 6) 0.001 7 dec)
!(mutateRule (AND terminal 4) 0.7)
!(addTerminalsToSpace (x y z a) &grammarSpace)
!(initializeStochasticSpace &alphabetSpace &grammarSpace &stochasticGrammarSpace)
!(extractAtomSpace &stochasticGrammarSpace)
!(calculateProbability (junctor OR) &stochasticGrammarSpace)
!(getProdRule (AND x) &stochasticGrammarSpace)
!(generateRelations (AND (OR x y) x) NONE ()) 
!(extractAtomSpace &grammarSpace)
;; !(== () (collapse (match &stochasticGrammarSpace (x $rhs $value) $rhs)))
!(generateInstances junctor &grammarSpace &stochasticGrammarSpace 4)
;; !(let $result (/ 1 3) $result)
;; !(initializeStochasticSpace &grammarSpace)
