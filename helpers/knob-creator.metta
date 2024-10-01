(= (reverse $list $accum)
    (if (== $list ())
        $accum
        (let $head (car-atom $list)
            (reverse (cdr-atom $list) (cons-atom $head $accum))
        )
    )
)

(=(decorate-with-knobs $exp $accum)
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
                            ($newAccum (cons-atom (OR AND) $accum))
                            ($reversedAccum (reverse $newAccum ()))
                        )
                        (decorate-with-knobs $tail $reversedAccum)
                    )
                    (if $isHeadExpression
                        (let* ( 
                                ($resultOfHead (decorate-with-knobs $head ()))
                                ($accumulatedTail (cons-atom $resultOfHead $accum))
                                ($finalAccumulator (reverse $accumulatedTail ()))
                            )
                            (decorate-with-knobs $tail $finalAccumulator)
                        )
                        (let* (
                                ($tailKnob (cons-atom ($head (NOT $head) subexp) $accum))
                                ($reversedKnob (reverse $tailKnob ()))
                            )
                            (decorate-with-knobs $tail $reversedKnob)
                        )
                    )


                )
            )
             (reverse (cons-atom ($exp (NOT $exp) subexp) $accum) ())
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
!(decorate-with-knobs (AND (OR x1 x2) x3) ())


;;!(reverse-atom (1 2 3 4 5) ())