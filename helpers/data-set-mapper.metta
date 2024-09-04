(=(mapper $exp $setInstance)
    (case $exp(
        (($op $exp1 $exp2)
                (let* (
                        ($type1 (get-metatype $exp1))
                        ($type2 (get-metatype $exp2))
                        ($isType1Expression (== $type1 Expression))
                        ($isType2Expression (== $type2 Expression))
                    )
                    (if (and $isType1Expression $isType2Expression)
                        ($op (mapper $exp1 $setInstance) (mapper $exp2 $setInstance))
                        (if $isType1Expression 
                            ($op (mapper $exp1 $setInstance) )
                        )
                    )
                )
            )

        )
    )

)
(=(mapSingleInstance $value Nil)
    NONE
)
(=(mapSingleInstance $value (Cons $x $xs))
    (case $x(
        (($var $boolVal) 
            (if (or (== $var result) (== $var Result))
                NONE
                (if (== $var $value)
                    $boolVal
                    (mapSingleInstance $value $xs)
                )
            )
        )
        ($_ NONE)
    
    ))
    
)