;;!(register-module )
;; !(import &self data-set-mapper)
;; !(let $x (superpose (1 2 3 4 5 6)) $x)
;; !(superpose (1 2 4 5 6))
!(bind! random (py-atom random.randint))
;; The below function split the atom into two points.
;; k is an index where the list is split
(= (length $atoms)
        (if (== $atoms ())
                0
                (+ 1 (length (cdr-atom $atoms)))
        )
)
;; splits the atom/llist at index x given x  
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
                ($splitPoint (random 0 (- $len 1)))
                (($fir $las) (split-at-pointK $atoms $splitPoint ()))
              )
                (collapse (union (superpose $las) (superpose $fir)))
        )
)
;; applies randomization to the list probabilistically
(= (randomizeList $atoms)
        (if (isEven (random 0 10))
                (random-shuffleK $atoms)
                $atoms
        )
)
;; checks for an availability of an even number.

(= (isEven $number)
        (if (== (% $number 2) 0)
                True 
                False
        )
)
!(randomizeList (a b c d e f g h i j k))
;; (= (iden-func $x) (/ $x 1))
;; (= (randomize $f $data)
;;         (collapse ($f (superpose $data)))
;; )
!(length (1 2 3 4 5 6))
!(split-at-pointK (a b c d e f g) 3 ())
!(randomize iden-func (1 2 3 4 5 6))
;; (if (>= (let* 
;;     (
;;         ($weight#1368 (match GroundingSpace-0x573ac496e238 
;;                     ((junctor junctor 7) (junctor OR 7) $value#1369) $value#1369))
;;     ) 
;;         (/ $weight#1368 0)
;;     ) 0.28731513333818826) 
        ;; (if (== (junctor junctor 7) junctor) 
        ;;     (if (>= (random.random) 0.5) 
        ;;         (AND (exploitInstances (junctor OR 7) GroundingSpace-0x573ac498ac08 GroundingSpace-0x573ac496e238 2 (- 3 1)) 
        ;;             (exploitInstances (junctor OR 7) GroundingSpace-0x573ac498ac08 GroundingSpace-0x573ac496e238 2 (- 3 1))) 
        ;;             (OR (exploitInstances (junctor OR 7) GroundingSpace-0x573ac498ac08 GroundingSpace-0x573ac496e238 2 (- 3 1)) 
        ;;             (exploitInstances (junctor OR 7) GroundingSpace-0x573ac498ac08 GroundingSpace-0x573ac496e238 2 (- 3 1)))) 
        ;;             (if (isAtomJunctor (junctor junctor 7)) 
        ;;                 ((junctor junctor 7) 
        ;;                     (exploitInstances (junctor OR 7) GroundingSpace-0x573ac498ac08 GroundingSpace-0x573ac496e238 2 (- 3 1)) 
        ;;                     (exploitInstances (junctor OR 7) GroundingSpace-0x573ac498ac08 GroundingSpace-0x573ac496e238 2 (- 3 1))) 
        ;;                     (exploitInstances (junctor OR 7) GroundingSpace-0x573ac498ac08 GroundingSpace-0x573ac496e238 2 (- 3 1)))) 
        ;;                     (exploitInstances (junctor OR 7) GroundingSpace-0x573ac498ac08 GroundingSpace-0x573ac496e238 2 (- 3 1)))
;;    !(collapse (let ($a $b $c) (superpose ((a b c) (d e f) (g h i))) $a)) ;; non-deterministic hell

;; !(mapper (OR (AND A (OR A B)) B) (Cons (A True) (Cons (B False) Nil)))
;; (if (cond) (truth value))
;; (
;;         (if (> $depth#39732 0) 
;;         (let $admissibleDerivs#39733 (filterProductionByLeftHandSide GroundingSpace-0x5907bb8f2ff8 $initSymbol#39728) (if (== $admissibleDerivs#39733 ()) $initSymbol#39728 (let* (($availableRules#39734 (findSuitableProdRules $initSymbol#39728 GroundingSpace-0x5907bbb27ec8)) ($randomizedRules#39735 (randomizeList $availableRules#39734)) ($kRules#39736 (selectK $randomizedRules#39735 1 ())) (($lhs#39737 $rhs#39738 $weight#39739) (superpose $kRules#39736)) ($calculatedProbability#39740 (calculateProbability ($lhs#39737 $rhs#39738) GroundingSpace-0x5907bbb27ec8))) (if (>= $calculatedProbability#39740 (random.random)) (let* (($subTreeDerivationLeft#39741 (exploitInstances $rhs#39738 GroundingSpace-0x5907bb8f2ff8 GroundingSpace-0x5907bbb27ec8 1 (- $depth#39732 1))) ($subTreeDerivationRight#39742 (exploitInstances $rhs#39738 GroundingSpace-0x5907bb8f2ff8 GroundingSpace-0x5907bbb27ec8 1 (- $depth#39732 1)))) (if (== $lhs#39737 junctor) (if (>= (random.random) 0.5) (AND $subTreeDerivationLeft#39741 $subTreeDerivationRight#39742) (OR $subTreeDerivationLeft#39741 $subTreeDerivationRight#39742)) (if (isAtomJunctor $lhs#39737) ($lhs#39737 $subTreeDerivationLeft#39741 $subTreeDerivationRight#39742) $subTreeDerivationLeft#39741))) (exploitInstances $rhs#39738 GroundingSpace-0x5907bb8f2ff8 GroundingSpace-0x5907bbb27ec8 1 (- $depth#39732 1)))))) (match GroundingSpace-0x5907bbb27ec8 (terminal $value#39743 $score#39744) $value#39743)))

(
        (let* (
                ($head#1718 (generateTrees expr GroundingSpace-0x63c67a6811f8 GroundingSpace-0x63c67a8b5bc8 3 (- 1 1))) 
                ($rhs'#1719 (if (== (get-metatype (uop expr)) Expression) (car-atom (uop expr)) (uop expr))) 
                ($left#1720 (generateTrees $rhs'#1719 GroundingSpace-0x63c67a6811f8 GroundingSpace-0x63c67a8b5bc8 3 (- 1 1))) 
                ($right#1721 (generateTrees $rhs'#1719 GroundingSpace-0x63c67a6811f8 GroundingSpace-0x63c67a8b5bc8 3 (- 1 1)))) 
                (if (== expr bop) 
                        ($head#1718 $left#1720 $right#1721) 
                        ($head#1718 $left#1720)
                )
        ) 
                (let* (
                        ($head#2216 (generateTrees bop GroundingSpace-0x63c67a6811f8 GroundingSpace-0x63c67a8b5bc8 3 (- 1 1)))
                        ($rhs'#2217 (if (== (get-metatype OR) Expression) (car-atom OR) OR)) 
                        ($left#2218 (generateTrees $rhs'#2217 GroundingSpace-0x63c67a6811f8 GroundingSpace-0x63c67a8b5bc8 3 (- 1 1))) ($right#2219 (generateTrees $rhs'#2217 GroundingSpace-0x63c67a6811f8 GroundingSpace-0x63c67a8b5bc8 3 (- 1 1)))) (if (== bop bop) ($head#2216 $left#2218 $right#2219) ($head#2216 $left#2218)))), ((let* (($head#1718 (generateTrees expr GroundingSpace-0x63c67a6811f8 GroundingSpace-0x63c67a8b5bc8 3 (- 1 1))) ($rhs'#1719 (if (== (get-metatype term) Expression) (car-atom term) term)) ($left#1720 (generateTrees $rhs'#1719 GroundingSpace-0x63c67a6811f8 GroundingSpace-0x63c67a8b5bc8 3 (- 1 1))) ($right#1721 (generateTrees $rhs'#1719 GroundingSpace-0x63c67a6811f8 GroundingSpace-0x63c67a8b5bc8 3 (- 1 1)))) (if (== expr bop) ($head#1718 $left#1720 $right#1721) ($head#1718 $left#1720))) (let* (($head#3708 (generateTrees bop GroundingSpace-0x63c67a6811f8 GroundingSpace-0x63c67a8b5bc8 3 (- 1 1))) ($rhs'#3709 (if (== (get-metatype OR) Expression) (car-atom OR) OR)) ($left#3710 (generateTrees $rhs'#3709 GroundingSpace-0x63c67a6811f8 GroundingSpace-0x63c67a8b5bc8 3 (- 1 1))) ($right#3711 (generateTrees $rhs'#3709 GroundingSpace-0x63c67a6811f8 GroundingSpace-0x63c67a8b5bc8 3 (- 1 1)))) (if (== bop bop) ($head#3708 $left#3710 $right#3711) ($head#3708 $left#3710)))), ((let* (($head#1718 (generateTrees expr GroundingSpace-0x63c67a6811f8 GroundingSpace-0x63c67a8b5bc8 3 (- 1 1))) ($rhs'#1719 (if (== (get-metatype term) Expression) (car-atom term) term)) ($left#1720 (generateTrees $rhs'#1719 GroundingSpace-0x63c67a6811f8 GroundingSpace-0x63c67a8b5bc8 3 (- 1 1))) ($right#1721 (generateTrees $rhs'#1719 GroundingSpace-0x63c67a6811f8 GroundingSpace-0x63c67a8b5bc8 3 (- 1 1)))) (if (== expr bop) ($head#1718 $left#1720 $right#1721) ($head#1718 $left#1720))) (let* (($head#4326 (generateTrees bop GroundingSpace-0x63c67a6811f8 GroundingSpace-0x63c67a8b5bc8 3 (- 1 1))) ($rhs'#4327 (if (== (get-metatype OR) Expression) (car-atom OR) OR)) ($left#4328 (generateTrees $rhs'#4327 GroundingSpace-0x63c67a6811f8 GroundingSpace-0x63c67a8b5bc8 3 (- 1 1))) ($right#4329 (generateTrees $rhs'#4327 GroundingSpace-0x63c67a6811f8 GroundingSpace-0x63c67a8b5bc8 3 (- 1 1)))) (if (== bop bop) ($head#4326 $left#4328 $right#4329) ($head#4326 $left#4328))))