;; Example of program synthesis using the backward chainer (operators
;; are not curried).

;; Define Nat
(: Nat Type)
(: Z Nat)
(: S (-> Nat Nat))

;; Knowledge base
!(bind! &kb (new-space))
!(add-atom &kb (: AND (-> Boolean Boolean Boolean)))
!(add-atom &kb (: OR (-> Boolean Boolean Boolean)))
!(add-atom &kb (: NOT (-> Boolean Boolean)))
!(add-atom &kb (: X Boolean))
!(add-atom &kb (: Y Boolean))

;; Backward chainer
(: bc (-> $a                            ; Knowledge base space
          Nat                           ; Maximum depth
          $b                            ; Query
          $b))                          ; Result
;; Base case
(= (bc $kb $_ (: $prg $type)) (match $kb (: $prg $type) (: $prg $type)))
;; Recursive steps
;; Unary operators
(= (bc $kb (S $k) (: ($fun $arg) $type))
   (let* (((: $fun (-> $prms $type)) (bc $kb $k (: $fun (-> $prms $type))))
          ((: $arg $prms) (bc $kb $k (: $arg $prms))))
     (: ($fun $arg) $type)))
;; Binary operators
(= (bc $kb (S $k) (: ($fun $arg1 $arg2) $type))
   (let* (
     ((: $fun (-> $prms1 $prms2 $type))
           (bc $kb $k (: $fun (-> $prms1 $prms2 $type))))
          ((: $arg1 $prms1) (bc $kb $k (: $arg1 $prms1)))
          ((: $arg2 $prms2) (bc $kb $k (: $arg2 $prms2)))
     )
     (: ($fun $arg1 $arg2) $type)

     )
)

;; Test bc
!(bc &kb Z (: $prg Boolean))
!(bc &kb (S Z) (: $prg Boolean))
!(bc &kb (S (S Z)) (: $prg Boolean))
