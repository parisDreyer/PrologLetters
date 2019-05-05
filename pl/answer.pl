:- consult("utilities/list_operations").
:- consult("utilities/relations").
:- consult("eliza").
:- consult("data").

% max(X,Y,Z) :-
%     (  X =< Y
%     -> Z = Y
%     ;  Z = X
%      ).

answer(Question, Explanation) :-
  (
    try_explain(Question, Output)
    -> Explanation = Output
    ;
    log_to_data(Question, Explanation)
  ).

try_explain(Question, Explanation) :-
  already_logged(Question, Out),
  flatten(Out, Flat),
  atomize(Flat, Logic),
  eliza_questions(Question, Logic, Explanation).

eliza_questions(Question, Logic, Explanation) :-
  eliza(Question, ElizaOutput1),
  eliza(Logic, ElizaOutput2),
  append(ElizaOutput1, ElizaOutput2, ElizaOutputs),
  append(ElizaOutputs, [], Explanation).

log_to_data(Question, Explanation) :-
  log_to_data(Question),
  atomize(Question, Explanation).

log_to_data(Question) :-
    open('data.pl', append, Stream),
      write(Stream,  "data("),
        write(Stream, Question),
        write(Stream, ').'),
      nl(Stream),
      close(Stream),
    consult("data").

already_logged([], Explanation) :-
  Explanation = [],
  fail.
already_logged([H|T], Explanation) :-
  (
    is_data(H, ExplanationItem)
    -> Explanation = ExplanationItem
    ;
    already_logged(T, Explanation)
  ).


data(X, Y) :- data(X), Y = X.
:- multifile data/1.
data([""]).

is_data(X, Explanation) :-
  data(Y, Explanation),
  member(X, Y).

is_data(X, Explanation) :-
  data(Y, Explanation),
  subset_with_replacement(X, Y).
