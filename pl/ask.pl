:- consult("./utilities/relations").
:- consult("./utilities/list_operations").
:- consult("./answer").

ask(Question, Explanation) :-
  atomize(Atomized, Question),
  answer(Atomized, Explanation).
