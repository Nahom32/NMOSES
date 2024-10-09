;;representation method 1
(x1 (x2 (x3 x4) x6)) ;; x1 -> x2->x3 
                     ;;        |-> x4
                     ;;  | ----> x6
(x1 x2 (x3) x4)
;;representation method 2
(= (import $y as $x) 
    (bind! $x (py-atom $y)))
(= (import $x) 
    (import $x as $x))


;;!(import normal_distribution as normal)
;;!(import! &self normal_distribution)
!(import normal_distribution.normal)
!(normal_distribution.normal 0.6 0.8 0.7)
;;!(normal 0.3 0.2 0.1)



