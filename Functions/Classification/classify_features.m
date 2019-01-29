clear acc acc1 acc2 X_added 
for i=1:3

    switch(i)
        case 1
            [acc1, acc2]= Apply_LeavOut_classification(X, Y);
            acc=[acc1,acc2];
        
        case 2
        for j=2:5
            switch(j)

                case 2
                X_added=X_FT;

                case 3
                clear X_added
                X_added=Max_X_FT;

                case 4
                X_added(:,end+1)=I;

                case 5
                X_added(:,end+1)=X_DC;
                
            end
            
           clear X_fourier
           X_fourier= X_added;
           [acc1, acc2]= Apply_LeavOut_classification(X_fourier, Y);
           acc(j,1:2)=[acc1,acc2];
         end
        
        case 3
            X_added= ESD;
            X_ESD=X_added;
           [acc1, acc2]= Apply_LeavOut_classification(X_ESD, Y);
           acc(end+1,1:2)=[acc1,acc2];
            
%         case 4
%             X_added= wavelet_features;
%             X_wavelet= X_added;
%            [acc1, acc2]= Apply_LeavOut_classification(X_wavelet, Y);
%            acc(end+1,1:2)=[acc1,acc2];
           
%         case 5
%         
%          for j=8:12
%             switch(j)
% 
%                 case 8
%                 clear X_added
%                 X_added=F_featuresA_h1;
% 
%                 case 9
%                 clear X_added
%                 X_added=S_featuresA_h1;
% 
%                 case 10
%                 clear X_added
%                 X_added=B_featuresA_h1;
% 
%                 case 11
%                 clear X_added
%                 X_added=P_featuresA_h1;
% 
%                 case 12
%                 clear X_added
%                 X_added= AF_featuresA_h1;
%         
%             end
%             
%            clear X_SCSA
%            X_SCSA=X_added;
%            [acc1, acc2]= Apply_LeavOut_classification(X_SCSA, Y);
%            acc(j,1:2)=[acc1,acc2];
%              
%          end
         
    end
end