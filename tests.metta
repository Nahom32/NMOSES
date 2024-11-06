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