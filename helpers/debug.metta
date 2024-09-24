(() (() (() (() (x x (OR x) (AND x) x) (OR (x)) (AND (x)) x) (OR (x)) (AND (x)) x) (OR (x)) (AND (x)) x) (OR (x)) (AND (x)) x)
((((terminal-val) (terminal-val) AND) (NOT terminal-val) ((terminal-val) (terminal-val) OR) terminal-val (((terminal-val) (terminal-val) AND) (NOT terminal-val) ((terminal-val) (terminal-val) OR) terminal-val (((terminal-val) (terminal-val) AND) (NOT terminal-val) ((terminal-val) (terminal-val) OR) terminal-val ((terminal-val terminal-val AND) (NOT terminal-val) (terminal-val terminal-val OR) terminal-val terminal-val)))) terminal-val ((terminal-val) (terminal-val) OR) (NOT terminal-val) ((terminal-val) (terminal-val) AND))
(
    (AND (terminal-val) (terminal-val)) 
    (OR (terminal-val) (terminal-val)) 
    ((AND (terminal-val) (terminal-val)) (OR (terminal-val) (terminal-val)) ((AND (terminal-val) (terminal-val)) (OR (terminal-val) (terminal-val)) ((AND (terminal-val) (terminal-val)) (OR (terminal-val) (terminal-val)) ((AND terminal-val terminal-val) (OR terminal-val terminal-val) terminal-val terminal-val (NOT terminal-val)) terminal-val (NOT terminal-val)) terminal-val (NOT terminal-val)) terminal-val (NOT terminal-val)) 
    terminal-val (NOT terminal-val)
)
(OR (OR terminal-val terminal-val) (AND (AND terminal-val terminal-val) (AND terminal-val terminal-val)))