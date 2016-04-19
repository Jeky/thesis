PATH = '~/data/thesis/sample-final/';
OUT = '~/Desktop/final work/thesis/figures/';
NAMES = {...
    'Suspended Users',...
    'Non-Suspended Users',...
};
BIN_N = 10;

SYMBOLS = {...
    'xr',...
    'ob'
};

% 
% % tweet count distribution
% disp('tweet count distribution');
% FILES = {'suspended-tweet-count.txt.dist',...
%          'non-suspended-tweet-count.txt.dist'};
%          
% for i = 1:2
%     in = sprintf('%s%s', PATH, char(FILES(i)));
%     data = load(in);
%     average = sum(data(:,1) .* data(:,2)) / sum(data(:,2))
%     loglog(data(:,1), data(:,2), char(SYMBOLS(i)), 'DisplayName', char(NAMES(i)));
%     xlabel('# Tweets');
%     ylabel('# Users');
%     hold all;
%     
% end
% legend('show');
% saveas(gcf, sprintf('%s%s.eps', OUT, 'tweet-count-dist'), 'epsc2');
% close(gcf);
% 
% 
% % token per user distribution
% disp('tweet length distribution');
% FILES = {'suspended-token-per-user-count.txt.dist',...
%          'non-suspended-token-per-user-count.txt.dist'};
%          
% for i = 1:2
%     in = sprintf('%s%s', PATH, char(FILES(i)));
%     data = load(in);
%     average = sum(data(:,1) .* data(:,2)) / sum(data(:,2))
%     loglog(data(:,1), data(:,2), char(SYMBOLS(i)), 'DisplayName', char(NAMES(i)));
%     xlabel('# Tokens');
%     ylabel('# Users');
%     hold all;
%     
% end
% legend('show');
% saveas(gcf, sprintf('%s%s.eps', OUT, 'token-per-user-count-dist'), 'epsc2');
% close(gcf);



% 
% % token distribution
% disp('token distribution');
% FILES = {'suspended-token-freq.txt.dist',...
%          'non-suspended-token-freq.txt.dist'};
% 
% for i = 1:2
%     in = sprintf('%s%s', PATH, char(FILES(i)));
%     data = load(in);
%     unique_count = sum(data(:,2))
%     total = sum(data(:,1) .* data(:,2))
%     average = sum(data(:,1) .* data(:,2)) / sum(data(:,2))
%     loglog(data(:,1), data(:,2), '.');
%     xlabel('Token Frequency');
%     ylabel('# Token');
%     
%     saveas(gcf, sprintf('%s%s.eps', OUT, get_filename(char(FILES(i)))), 'epsc2');
%     close(gcf);
% end
% 
% % tweet len distribution
% disp('tweet len distribution');
% FILES = {'suspended-tweet-len.txt.dist',...
%          'non-suspended-tweet-len.txt.dist'};
%          
% for i = 1:2
%     in = sprintf('%s%s', PATH, char(FILES(i)));
%     data = load(in);
%     average = sum(data(:,1) .* data(:,2)) / sum(data(:,2))
%     
%     sorted = sortrows(data);
%     plot(sorted(:,1), sorted(:,2), '-', 'DisplayName', char(NAMES(i)));
%     hold all;
% end
% 
% xlabel('# Tokens');
% ylabel('# Tweets');
% legend('show');
% xlim([0,35]);
% saveas(gcf, sprintf('%stweet-len-dist.eps', OUT), 'epsc2');
% close(gcf);
%     
% 
% % retweet distribution
% disp('retweet distribution');
% FILES = {'suspended-retweet-rate.txt',...
%          'non-suspended-retweet-rate.txt'};
%          
% for i = 1:2
%     in = sprintf('%s%s', PATH, char(FILES(i)));
%     data = load(in);
%     average = mean(data)
%     
%     intData = floor(data * BIN_N);
%     bin = ones(BIN_N, 1);
%     j = 1;
%     
%     for count = 0:BIN_N
%         bin(j) = length(find(intData == count)) / length(data);
%         if bin(j) == 0
%             bin(j) = 1;
%         end
%         j = j + 1;
%     end
%     
%     bin
%     
%     semilogy(0:1/BIN_N:1, bin, 'DisplayName', NAMES(i));
%     hold all;
% end
% 
% xlabel('Probability of Retweet');
% ylabel('User Percentage');
% legend('show');
% saveas(gcf, sprintf('%srt-dist.eps', OUT), 'epsc2');
% close(gcf);
% 
% 
% 
% % url distribution
% disp('url distribution');
% FILES = {'suspended-url-rate.txt',...
%          'non-suspended-url-rate.txt'};
%          
% for i = 1:2
%     in = sprintf('%s%s', PATH, char(FILES(i)));
%     data = load(in);
%     average = mean(data)
%     
%     intData = floor(data * BIN_N);
%     bin = ones(BIN_N, 1);
%     j = 1;
%     
%     for count = 0:BIN_N
%         bin(j) = length(find(intData == count)) / length(data);
%         if bin(j) == 0
%             bin(j) = 1;
%         end
%         j = j + 1;
%     end
%     
%     bin
%     
%     semilogy(0:1/BIN_N:1, bin, 'DisplayName', NAMES(i));
%     hold all;
% end
% 
% xlabel('Probability of URL');
% ylabel('User Percentage');
% legend('show');
% saveas(gcf, sprintf('%surl-dist.eps', OUT), 'epsc2');
% close(gcf);
% 
% 
% % mention distribution
% disp('mention distribution');
% FILES = {'suspended-mention-rate.txt',...
%          'non-suspended-mention-rate.txt'};
%          
% for i = 1:2
%     in = sprintf('%s%s', PATH, char(FILES(i)));
%     data = load(in);
%     average = mean(data)
%     
%     intData = floor(data * BIN_N);
%     bin = ones(BIN_N, 1);
%     j = 1;
%     
%     for count = 0:BIN_N
%         bin(j) = length(find(intData == count)) / length(data);
%         if bin(j) == 0
%             bin(j) = 1;
%         end
%         j = j + 1;
%     end
%     
%     bin
%     
%     semilogy(0:1/BIN_N:1, bin, 'DisplayName', NAMES(i));
%     hold all;
% end
% 
% xlabel('Probability of Mention');
% ylabel('User Percentage');
% legend('show');
% saveas(gcf, sprintf('%smention-dist.eps', OUT), 'epsc2');
% close(gcf);
% 
% 
% % hashtag distribution
% disp('hashtag distribution');
% FILES = {'suspended-hashtag-rate.txt',...
%          'non-suspended-hashtag-rate.txt'};
%          
% for i = 1:2
%     in = sprintf('%s%s', PATH, char(FILES(i)));
%     data = load(in);
%     average = mean(data)
%     
%     intData = floor(data * BIN_N);
%     bin = ones(BIN_N, 1);
%     j = 1;
%     
%     for count = 0:BIN_N
%         bin(j) = length(find(intData == count)) / length(data);
%         if bin(j) == 0
%             bin(j) = 1;
%         end
%         j = j + 1;
%     end
%     
%     bin
%     
%     semilogy(0:1/BIN_N:1, bin, 'DisplayName', NAMES(i));
%     hold all;
% end
% 
% xlabel('Probability of Hashtag');
% ylabel('User Percentage');
% legend('show');
% saveas(gcf, sprintf('%shashtag-dist.eps', OUT), 'epsc2');
% close(gcf);


% 
% % feature-acc-rel
% disp('feature-acc-rel');
% FILES = {'mi-test.txt',...
%          'tokenmi-test.txt'};
% M_NAMES = {'MI', 'TMI'};     
% SYMBOLS = {'x-', 'o--'};
% for i = 1:2
%     in = sprintf('%s%s', PATH, char(FILES(i)));
%     data = load(in);
%     acc = (data(:,2) + data(:,5)) ./ (data(:,2) + data(:,3)+data(:,4)+data(:,5))
%     semilogx(data(:,1), acc, char(SYMBOLS(i)), 'DisplayName', M_NAMES(i));
%     hold all;
% end
% 
% xlabel('Feature Size');
% ylabel('Accuracy');
% legend('show', 'Location', 'southeast');
% saveas(gcf, sprintf('%sfeature-acc-rel.eps', OUT), 'epsc2');
% close(gcf);
% 
% 
% % feature-f1-rel
% disp('feature-f1-rel');  
% for i = 1:2
%     in = sprintf('%s%s', PATH, char(FILES(i)));
%     data = load(in);
%     f1 = 2 * data(:,2) ./ (2 * data(:,2) + data(:,3) + data(:,4))
%     semilogx(data(:,1), f1, char(SYMBOLS(i)), 'DisplayName', M_NAMES(i));
%     hold all;
% end
% 
% xlabel('Feature Size');
% ylabel('F1');
% legend('show', 'Location', 'southeast');
% saveas(gcf, sprintf('%sfeature-f1-rel.eps', OUT), 'epsc2');
% close(gcf);
% 
% user matrix
disp('user matrix');
S = 11326;
NS = 11479;

    in = sprintf('%suv-tsne.out.txt', PATH);
    data = load(in);
    plot(data(1:S, 1), data(1:S, 2), '.r', 'MarkerSize', 2);
    hold all;
    plot(data(S:NS+S, 1), data(S:NS+S, 2), '.b', 'MarkerSize', 2);

    lh = legend('Suspended Users', 'Normal Users');
    ch = get(lh,'Children');
    texth = ch(strcmp(get(ch, 'Type'), 'text'));
    set(texth, {'Color'}, {'b';'r'});
    
    saveas(gcf, sprintf('%suser-uv-matrix.eps', OUT), 'epsc2');
%     close(gcf);
