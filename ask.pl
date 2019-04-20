:- consult("./utilities/relations").
:- consult("./utilities/list_operations").
:- consult("./answer").

ask(Question) :-
  [H|T] = Question,
  H,T,
  answer(Question).
