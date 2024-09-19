(=(function-set-generator $type)
    (if (== $type boolean)
        (OR AND NOT)
        (if (== $type arithmetic)
            (ADD MULT DIVIDE SUB)
            NONE
        )
    )
)
!(function-set-generator boolean)


