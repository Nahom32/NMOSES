;; Example of program synthesis using the backward chainer (operators
;; are curried).

;; Define Nat
(: Nat Type)
(: Z Nat)
(: S (-> Nat Nat))

;; Knowledge base
!(bind! &kb (new-space))
!(add-atom &kb (: AND (-> Boolean (-> Boolean Boolean))))
!(add-atom &kb (: OR (-> Boolean (-> Boolean Boolean))))
!(add-atom &kb (: NOT (-> Boolean Boolean)))
!(add-atom &kb (: X Boolean))
!(add-atom &kb (: Y Boolean))

;; Backward chainer
(: bc (-> $a                            ; Knowledge base space
          Nat                           ; Maximum depth
          $b                            ; Query
          $b))                          ; Result
;; Base case
(= (bc $kb $_ (: $prf $ccln)) (match $kb (: $prf $ccln) (: $prf $ccln)))
;; Recursive step
(= (bc $kb (S $k) (: ($prfabs $prfarg) $ccln))
   (let* (((: $prfabs (-> $prms $ccln)) (bc $kb $k (: $prfabs (-> $prms $ccln))))
          ((: $prfarg $prms) (bc $kb $k (: $prfarg $prms))))
     (: ($prfabs $prfarg) $ccln)))

;; Test bc
!(bc &kb Z (: $prg Boolean))
!(bc &kb (S Z) (: $prg Boolean))
!(bc &kb (S (S Z)) (: $prg Boolean))
