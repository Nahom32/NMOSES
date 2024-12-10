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
(=(split-at-pointK $atoms $index $accum)
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