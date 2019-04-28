:- consult("./utilities/relations").
:- consult("./utilities/list_operations").
:- consult("./answer").

ask(Question) :-
  atomize(Atomized, Question),
  answer(Atomized).
