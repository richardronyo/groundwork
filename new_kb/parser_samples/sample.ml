type point = { x : int; y : int }

(* adds two points *)
let add (p1 : point) (p2 : point) : point =
  { x = p1.x + p2.x; y = p1.y + p2.y }

module Sample = struct
  let origin = { x = 0; y = 0 }
end

let () =
  let result = add Sample.origin { x = 3; y = 4 } in
  Printf.printf "%d, %d\n" result.x result.y
