(= (reverse $list $accum)
    (if (== $list ())
        $accum
        (let $head (car-atom $list)
            (reverse (cdr-atom $list) (cons-atom $head $accum))
        )
    )
)

(= (decorate-with-knobs $exp $accum)
    (if (== $exp ())
        $accum
        (if (== (get-metatype $exp) Expression)
            
            (let* (
                ($head (car-atom $exp))
                ($tail (cdr-atom $exp))
                ($isHeadJunctor (or (eq $head OR) (eq $head AND)))
                ($isHeadExpression (== (get-metatype $head) Expression))
            )
                (if $isHeadJunctor
                    (let*  (
                            ($modifiedAccum (append $accum (OR AND)))
                            
                        )
                        (decorate-with-knobs $tail $modifiedAccum)
                    )
                    (if $isHeadExpression
                        (let* ( 
                                ($resultOfHead (decorate-with-knobs $head ()))
                                ($accumulatedTail (append $accum $resultOfHead))
                                ;; ($finalAccumulator (reverse $accumulatedTail ()))
                            )
                            (decorate-with-knobs $tail $accumulatedTail)
                        )
                        (let* (
                                ($tailKnob  (append $accum ($head (NOT $head) subexp)))
                                ;; ($reversedKnob (reverse $tailKnob ()))
                            )
                            (decorate-with-knobs $tail $tailKnob)
                        )
                    )


                )
            )
             (append $accum  ($exp (NOT $exp) subexp))
        )

    )  
)
(=(eq $exp1 $exp2)
    (let* (
        ($typeExp1 (get-metatype $exp1))
        ($typeExp2 (get-metatype $exp2))

    )
        (if (== $typeExp1 $typeExp1)
            (== $exp1 $exp2)
            False
        )

    )
)
(= (append $exp1 $exp2)
    (union (superpose (toList $exp1)) (superpose $exp2))
)
(=(toList $exp)
    (if (== (get-metatype $exp) Expression)
        $exp
        ($exp)
    )
)
;; (= (append-helper $list $accum)
    
;;         (if (== $list ())
;;             $accum
;;             (let* (
;;                 ($head (car-atom $list))
;;                 ($tail (cdr-atom $list))
;;             )
;;                 (append-helper $tail (cons-atom $head $accum))
;;             )
;;         )

   
;; )
;; (= (append $list $data)
;;     (append-helper (reverse $list ()) ($data))
;; )
;;!(append (1 2 3 4 5 6) 7)
;; !(decorate-with-knobs (AND (OR x1 x2) x3) ())
!(decorate-with-knobs (OR (AND (OR x1 x2) x3) x4 x5) ())

(=(createInstances $representation $accum)
    (if (== () $representation)
        $accum
        (let* (
            ($head (car-atom $representation))
            ($tail (cdr-atom $representation))
            ($typeHead (get-metatype $head))
            ($typeTail (get-metatype $tail))
        )
            (if (== $typeHead Expression)
                (if (== $head (AND OR))
                    (let* (
                        ($andSubExp (createInstances $tail (AND)))
                        ($orSubExp (createInstances $tail (OR)))
                        ($andAccum (append $andSubExp $accum))
                        ($orAccum (append $orSubExp $accum))

                    )
                        $orAccum
                    )
                    (let $firstInstance (car-atom $head)
                        (if (== $firstInstance NOT)
                            (createInstances $tail (append $head $accum))
                            (let  $nextAccum (createInstances (superpose $head) $accum)
                                (createInstances $tail (append $nextAccum $tail))
                            )
                        )
                    )
                )
            (createInstances $tail (append $head $accum))
                
            )
        )
    )
    
)
;;
!(collapse (createInstances ((OR AND) ((OR AND) ((OR AND) (x1 (NOT x1) subexp) (x2 (NOT x2) subexp)) (x3 (NOT x3) subexp)) (x4 (NOT x4) subexp) (x5 (NOT x5) subexp)) ()))



;;!(reverse-atom (1 2 3 4 5) ())