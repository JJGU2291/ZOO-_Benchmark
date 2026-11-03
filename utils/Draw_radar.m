function [H] = Draw_radar(data, lim, labels, style)
    H={};
    n=length(data);
    adj_data=zeros(n,1);
    point=zeros(n,2);
    
    set(gca, 'units', 'normal', 'pos', [0.1 0.1 0.7 0.8]);
    axis off; axis equal; hold on;
    theta_last = pi/2;
    
    for i=1:n
        theta = 2*pi/n*i + pi/2;
        % 绘制轴线
        H{end+1} = plot([0, 500*cos(theta)], [0, 500*sin(theta)], 'k-', 'linewidth', 1.2);
        
        % 绘制背景虚线网格
        for j=1:5
           H{end+1} = plot([j*100*cos(theta_last), j*100*cos(theta)], ...
               [j*100*sin(theta_last), j*100*sin(theta)], '--', 'linewidth', 0.5, 'color', [0.7,0.7,0.7]);
        end
        theta_last = theta;
        
        % 动态映射数据到 0-500 坐标系
        if data(i) < lim(i,1)
            adj_data(i) = 0;
        elseif data(i) > lim(i,2)
            adj_data(i) = 500;
        else
            adj_data(i) = (data(i)-lim(i,1)) / (lim(i,2)-lim(i,1)) * 500;
        end
        point(i,1:2) = [adj_data(i)*cos(theta); adj_data(i)*sin(theta)];
        
        % 轴标签文字
        text_around(530*cos(theta), 530*sin(theta), labels{i}, theta, 11);
    end
    
    % 绘制动态刻度值
    for i=1:n
        theta = 2*pi/n*i + pi/2;
        for j=1:5
            val = lim(i,1) + (lim(i,2)-lim(i,1))/5*j;
            text_around(j*100*cos(theta), j*100*sin(theta), sprintf('%.1f', val), theta+pi/2, 7);
        end
    end
    
    % 绘制数据折线
    combined_style = [style.line style.marker];
    H{end+1} = plot([point(:,1); point(1,1)], [point(:,2); point(1,2)], ...
        combined_style, 'Color', style.color, 'LineWidth', 1.8, ...
        'MarkerFaceColor', style.color, 'MarkerSize', 5);
    
    axis tight;
end
 
function text_around(x,y,txt,theta,fontsize)
    if nargin==4
        fontsize=14;
    end
    section=mod(theta+pi/12,2*pi);
    if section>pi+pi/6
        %上对齐
        if section>1.5*pi+pi/6
            %左对齐
            text(x,y,txt,'Interpreter','latex','VerticalAlignment','cap','HorizontalAlignment','left','Fontsize',fontsize);
        elseif section>1.5*pi
            %中对齐
            text(x,y,txt,'Interpreter','latex','VerticalAlignment','cap','HorizontalAlignment','center','Fontsize',fontsize);
        else
            %右对齐
            text(x,y,txt,'Interpreter','latex','VerticalAlignment','cap','HorizontalAlignment','right','Fontsize',fontsize);
        end
    elseif section>pi
        %中、右对齐
        text(x,y,txt,'Interpreter','latex','VerticalAlignment','middle','HorizontalAlignment','right','Fontsize',fontsize);
    elseif section>pi/6
        %下对齐
        if section>0.5*pi+pi/6
            %右对齐
            text(x,y,txt,'Interpreter','latex','VerticalAlignment','bottom','HorizontalAlignment','right','Fontsize',fontsize);
        elseif section>0.5*pi
            %中对齐
            text(x,y,txt,'Interpreter','latex','VerticalAlignment','bottom','HorizontalAlignment','center','Fontsize',fontsize);
        else
            %左对齐
            text(x,y,txt,'Interpreter','latex','VerticalAlignment','bottom','HorizontalAlignment','left','Fontsize',fontsize);
        end
    else
        %中、左对齐
        text(x,y,txt,'Interpreter','latex','VerticalAlignment','middle','HorizontalAlignment','left','Fontsize',fontsize);
    end
end