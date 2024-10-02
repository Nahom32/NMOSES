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
(= (append-helper $list $accum)
    
        (if (== $list ())
            $accum
            (let* (
                ($head (car-atom $list))
                ($tail (cdr-atom $list))
            )
                (append-helper $tail (cons-atom $head $accum))
            )
        )

   
)
(= (append $list $data)
    (append-helper (reverse $list ()) ($data))
)
;;!(append (1 2 3 4 5 6) 7)
!(decorate-with-knobs (AND (OR x1 x2) x3) ())


;;!(reverse-atom (1 2 3 4 5) ())