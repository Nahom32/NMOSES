;; !(collapse (union (superpose (1 2 3 4 5)) (superpose (a b c d e))))
;; !(superpose (1))
;!(/ 1 3)
; ! (bind! py-rnd (py-atom random))
 ; ! (bind! randint (py-dot py-rnd randint))

 ;;(: random-bool (-> Boolean))

;; ! (random-bool 0 1)
;; ! (random-bool 0 1)
;; ! (random-bool 0 1)
;; ! ( (random-bool 0 1) (random-bool 0 1))
(
    ((junctor junctor 7) (junctor OR 7) (junctor AND 7)) 
    ((AND terminal 6)) 
    ((OR terminal 7)) 
    ((junctor junctor 7) (junctor OR 7) (junctor AND 7)) 
    ((junctor junctor 7) (junctor OR 7) (junctor AND 7))
)

(
    ((junctor junctor 10) (junctor AND 10) (junctor OR 10)) 
    ((OR terminal 10)) 
    ((junctor junctor 10) (junctor AND 10) (junctor OR 10)) 
    ((junctor junctor 10) (junctor AND 10) (junctor OR 10)) 
    ((AND terminal 9))
)
;; ((if True (OR (generateInstancesHelper OR GroundingSpace-0x5a2854798678 (- 3 1)) (generateInstancesHelper OR GroundingSpace-0x5a2854798678 (- 3 1))) (generateInstancesHelper OR GroundingSpace-0x5a2854798678 (- 3 1))))
(goodTrees (collapse (filter >= fitnessWithdata (superpose ((AND x1 x1))) 0.5)) badTrees (collapse (filter < fitnessWithdata (superpose ((AND x1 x1))) 0.5)))
!(superpose ((AND x1 x1)))
!(collapse (empty))
(let* 
    (
        ($_#12543 (reinforceProd ((bop AND 9) (expr (bop expr expr) 37.19390363447943)) 0.01 15 GroundingSpace-0x5bfb63fc12f8)) 
        ($stocAtoms#12544 (collapse (get-atoms GroundingSpace-0x5bfb63fc12f8))) 
        (() (println! (stocAtoms $stocAtoms#12544 tail ((term x2 9) (term x1 9)  ((bop OR 9) (expr (bop expr expr) 37.19390363447943))))))
    ) 
        (reinforceTreeHelper ((term x2 9) (term x1 9) ((bop OR 9) (expr (bop expr expr) 37.19390363447943))) 0.01 15 GroundingSpace-0x5bfb63fc12f8)
)