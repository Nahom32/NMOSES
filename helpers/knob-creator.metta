(=(decorate-with-knobs $exp $accum)
    (if (eq $exp ())
        $accum
        (if (== (get-metatype $exp) Expression)
            
            (let* (
                ($head (car-atom $exp))
                ($tail (cdr-atom $exp))
                ($isHeadJunctor (or (eq $head OR) (eq $head AND)))
                ($isHeadExpression (== (get-metatype $head) Expression))
            )
                (if $isHeadJunctor
                    (let $newAccum (cons-atom (OR AND) $accum)
                        (decorate-with-knobs $tail  $newAccum )
                    )
                    (if $isHeadExpression
                        (let $resultOfHead (decorate-with-knobs $head ())
                            (decorate-with-knobs $tail (cons-atom $resultOfHead $accum))
                        )
                        (decorate-with-knobs $tail  (cons-atom ($head (NOT $head) subexp) $accum))
                    )


                )
            )
             (cons-atom ($exp (NOT $exp) subexp) $accum)
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

(= (reverse-atom $list $accum)
    (if (== $list ())
        $accum
        (let $head (car-atom $list)
            (reverse-atom (cdr-atom $list) (cons-atom $head $accum))
        )
    )
)