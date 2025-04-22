% rules.pl
% Simple logic-based recommendations based on model output

has_pneumonia(yes) :-
    write('The patient is likely to have pneumonia. Further medical attention is advised.').

has_pneumonia(no) :-
    write('The patient is likely healthy. Continue regular monitoring.').
