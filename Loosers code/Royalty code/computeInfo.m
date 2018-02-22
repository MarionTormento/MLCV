
function info = computeInfo(inputnode, outputnodes)
%% Entropy before
for i=1:3
    if ~isempty(inputnode((inputnode(:,3) == i) == 1,:))
        prob(i,1) = size(inputnode((inputnode(:,3) == i) == 1,:),1)/size(inputnode(:,1),1);
    else
        prob(i,1) = 1; 
    end
end
entBefore = sum(-prob.*log(prob),1);
clear prob

%% Entropy After
for j = 1:2
    for i = 1:3
        if ~isempty(outputnodes{j}((outputnodes{j}(:,3) == i) == 1,:))
            prob(i,j) = size(outputnodes{j}((outputnodes{j}(:,3) == i) == 1,:),1)/size(outputnodes{j}(:,1),1);
        else
           prob(i,j) = 1; 
        end
    end
end
prob_parent = [size(outputnodes{1}(:,1),1)/size(inputnode(:,1),1) size(outputnodes{2}(:,1),1)/size(inputnode(:,1),1)];
entAfter = sum(sum(-prob.*log(prob),1).*prob_parent,2); % Entropy Before
clear prob_parent
clear prob

%% Information gain
info = entBefore - entAfter;
clear entAfter
clear entBefore
end %compute the info gai
