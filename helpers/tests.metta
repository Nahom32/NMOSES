;;!(register-module )
;; !(import &self data-set-mapper)
!(let $x (superpose (1 2 3 4 5 6)) $x)
!(superpose (1 2 4 5 6))
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