%%  eliza(+Stimuli, -Response) is det.
%   @param  Stimuli is a list of atoms (words).
%   adapted from: @author Richard A. O'Keefe (The Craft of Prolog)
%   https://swish.swi-prolog.org/example/eliza.pl
%   addaptions written by Luke Dreyer


eliza(Stimuli, Response) :-
    template(InternalStimuli, InternalResponse),
    match(InternalStimuli, Stimuli),
    match(InternalResponse, Response),
    !.

template([s([i,am]),s(X)], [s([why,are,you]),s(X),w('?')]).
template([s([you,are]),s(X)], [s([why,am,i]),s(X),w('?')]).
template([w(i),s(X),w(you)], [s([why,do,you]),s(X),w(me),w('?')]).
template([w(you),s(X),s([i, are]),s(Y)], [s([why,are,you]),s(X),w(i), s(Y), w('?')]).
template([w(you),s(X),w(i), s(Y)], [s([why,do,you]),s(X),w(i), s(Y), w('?')]).
template([w(you),s(X),w(me)], [s([why, do, i]),s(X), w(you), w('?')]).
template([s(X), w(is), s(Y)], [s([why, is, there]), s(Y), s(X), w('?')]).
template([s(X), w(are), s(Y)], [s([why, are, there]), s(Y), s(X), w('?')]).
template([s([what, is]), w(_), w(name)], Out) :-
  choose([
    [s([i, am, terrible, with, names])],
    [s([i, am, sorry, my, name, recognition, subroutine, is, not, implemented, yet])]
  ], Out).
template([s(X)], Out) :-
  choose([
    [s([why, do, you, say]), s(X), w('?')],
    [s([what, do, you, mean, by]), s(X), w('?')],
    [s([could, you, explain, more, to, me, about]), s(X), w('?')],
    [s([i, do, not, know, enough, about, that, to, agree, or, disagree, with, you])]
  ], Out).

match([],[]).
match([Item|Items],[Word|Words]) :-
    match(Item, Items, Word, Words).

match(w(Word), Items, Word, Words) :-
    match(Items, Words).
match(s([Word|Seg]), Items, Word, Words0) :-
    append(Seg, Words1, Words0),
    match(Items, Words1).

%% https://stackoverflow.com/questions/2261238/random-items-in-prolog
%% choose(List, Elt) - chooses a random element
%% in List and unifies it with Elt.
choose([], []).
choose(List, Elt) :-
        length(List, Length),
        random(0, Length, Index),
        nth0(Index, List, Elt).

/** <examples>

?- eliza([i, am, very, hungry], Response).
?- eliza([i, love, you], Response).

*/
