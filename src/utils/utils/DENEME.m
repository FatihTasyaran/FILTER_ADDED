% AUTHOR: BEYZA CAVUSOGLU 11.09.2023
% GOSH REORDERING ALGORITHM 

close all

figure_size = [488, 342, 560, 420];  % [left, bottom, width, height]


inp_files = dir('input*.txt');

% Number of input and output files
num_files_inp =length(inp_files);


pool = parpool('local', num_files_inp); % Specify the number of cores you want to use

% Define a parfor loop to process input files in parallel
parfor ind = 1:num_files_inp
     tic
   
    count1 = 1;
    count2 = 1;
    count3 = 1;
    count4 = 1;
    count5 = 1;
    count6 = 1;
    % Construct input and output file names using the loop variable
    input_file_name = sprintf('input%d.txt', ind);
    output_file_prefix = sprintf('output%d_', ind);
    input_file_prefix = sprintf('input%d', ind);

    input_files = dir(input_file_name);
    output_files = dir([output_file_prefix, '*.txt']);

    % Number of input and output files
    num_files =length(output_files);

    alpha_num_arr = zeros (size(output_files));

  % Read the input graph data from input.txt
    file_name_input = input_file_name;
    data_input = dlmread(file_name_input);
    
 % Extract the edges from the input data
    edges_input = data_input(:, 1:2);

    % Get the unique vertices in the graph
    vertices = unique(edges_input(:));
   
  % Create a graph from the edges and label the nodes
    G_input = graph(edges_input(:, 1)+ 1 , edges_input(:, 2) +1 );
    G_input.Nodes.Name = arrayfun(@(x) num2str(x), vertices, 'UniformOutput', false);
    
    figure(ind)
 % Subplot for the original graph visualization
    subplot(6, 5, 1);
    p_input = plot(G_input, 'Layout', 'force', 'EdgeColor', 'b', 'MarkerSize', 3, 'NodeColor', 'r', 'NodeLabel', G_input.Nodes.Name);
    title(['Original(', file_name_input, ')']);

    % Create a tuple containing the original vertices and their embeddings
        vertices = unique(edges_input(:));
    
        
  %--------------------------------------------------------
        % ADJACENCY MATRIX ORIGINAL SPY 1
        % Subplot for the adjacency matrix heatmap
        subplot(6, 5, 2);

        % Initialize the adjacency matrix with zeros MANUALLY CREATE
         adjacency_matrix = zeros(length(vertices));

        
        % Set the values in the adjacency matrix based on the edges (undirected)
        for i = 1:size(edges_input, 1)
            vertex1 = edges_input(i, 1) ;
            vertex2 = edges_input(i, 2) ;
            adjacency_matrix(vertex1 + 1, vertex2 + 1) = 1;
            adjacency_matrix(vertex2 + 1, vertex1 + 1) = 1; % Ensure symmetry
        end
        

         % Subplot for the sparsity pattern of the adjacency matrix
        spy(adjacency_matrix);    
        title(['Adj (', file_name_input, ')']);
        xlabel('Vertex Index');
        ylabel('Vertex Index');

        % Customize colors for the spy plot
        colormap(gca, [1, 1, 1; 0, 0, 0]); % White for zeros, Black for non-zeros


        % Keep the x and y tick labels as the sorted vertex indices
        xticks(1:size(adjacency_matrix, 1));
        yticks(1:size(adjacency_matrix, 1));
        xticklabels(num2str(vertices(1:end) ));
        yticklabels(num2str(vertices(1:end) ));

       % yticklabels(num2str(vertices(end:-1:1))); % Reverse the order
        xlabel('Original Vertex Index');
        ylabel('Original Vertex Index');

        %--------------------------------------------------------

        subplot(6, 5, 3);
        % Calculate the Reverse Cuthill-McKee (RCM) ordering
        rcm_order = symrcm(adjacency_matrix);
        spy(adjacency_matrix(rcm_order, rcm_order));
        title(['Adj RCM']);
        xlabel('RCM VInd');
        ylabel('RCM VInd');

        % Customize colors for the spy plot
        colormap(gca, [1, 1, 1; 0, 0, 0]); % White for zeros, Black for non-zeros

        % Set tick positions and labels for the RCM-ordered vertex indices
        tick_positions = 1:size(adjacency_matrix, 1);
        tick_labels = num2str(rcm_order' - 1);

        % Adjust tick positions and labels for better readability 
        % For example, to show every 10th vertex index
        % tick_positions = 1:10:size(adjacency_matrix, 1);
        % tick_labels = num2str(rcm_order(tick_positions)');

        xticks(tick_positions);
        yticks(tick_positions);
        xticklabels(tick_labels);
        yticklabels(tick_labels);

        xtickangle(45); % Rotate tick labels for better visibility 
%--------------------------------------------------------

        subplot(6, 5, 4);
         % Calculate the Approximate Minimum Degree (AMD) ordering
        amd_order = amd(adjacency_matrix);
        spy(adjacency_matrix(amd_order, amd_order));
        title(['Adj AMD']);
        xlabel('AMD VInd');
        ylabel('AMD VInd');

          % Customize colors for the spy plot
        colormap(gca, [1, 1, 1; 0, 0, 0]); % White for zeros, Black for non-zeros

        % Set tick positions and labels for the RCM-ordered vertex indices
        tick_positions = 1:size(adjacency_matrix, 1);
        tick_labels = num2str(amd_order' - 1);

        % Adjust tick positions and labels for better readability
        % For example, to show every 10th vertex index
        % tick_positions = 1:10:size(adjacency_matrix, 1);
        % tick_labels = num2str(rcm_order(tick_positions)');

        xticks(tick_positions);
        yticks(tick_positions);
        xticklabels(tick_labels);
        yticklabels(tick_labels);

        xtickangle(45); % Rotate tick labels for better visibility

        %--------------------------------------------------------  
        f1  = figure;
        f2  = figure;
        f3  = figure;
        f4  = figure;
        f5  = figure;         
        f6  = figure;         
        
    for file_num = 1:num_files
 
        % Read the graph embedding data from output.txt
        file_name_output = output_files(file_num).name;
        data_output = dlmread(file_name_output);

        parts = strsplit(file_name_output, '_'); % Split by underscores
        output_num = str2double(parts{1}(7:end)); % Extract number after "output"
        pos_num = str2double(parts{2}(4:end));    % Extract number after "pos"
        alpha_num = str2double(parts{3}(6:end));  % Extract number after "alpha"
        alpha_num_arr = vertcat(alpha_num_arr, alpha_num);
        neg_num = str2double(parts{4}(4:end));    % Extract number after "neg"       
        lrate_num = str2double(parts{5}(3:end));    % Extract number after "lrate"       

        % Extract the wedge number by removing ".txt" from the last part
        sm_str = parts{6};
        sm_num = str2double(sm_str(3:end-4)); % Remove ".txt"


        % Extract the embedding data and dimensions from the output data
        num_vertices = data_output(1, 1);
        dimension = data_output(1, 2);
        embedding_data = data_output(2:end, 2:end); % Read only the embedding data
   
        vertex_embedding_tuple = [vertices, embedding_data];

        % Sort the tuple based on the second column (embedding values)
        [~, sorted_indices] = sort(vertex_embedding_tuple(1:end, 2));
        sorted_vertex_embedding_tuple = vertex_embedding_tuple(sorted_indices, :);

        % Get the sorted vertices and corresponding embedding values
        sorted_vertices = sorted_vertex_embedding_tuple(:, 1);
        sorted_embedding_data = sorted_vertex_embedding_tuple(:, 2:end);


        sorted_vertex_indices = sorted_vertex_embedding_tuple(:, 1);
        adjacency_matrix_sorted = adjacency_matrix(sorted_indices, sorted_indices);   
        
        %------------------------------------------------
        % 1
        % Check the conditions
        if (pos_num == 1 || pos_num == 2 || pos_num == 3) && ...
           (lrate_num == 0.025) && ...
           (alpha_num == 0 || alpha_num == 0.5 || alpha_num == 0.75) && ...
           (sm_num == 0 || sm_num == 1 || sm_num == 2) && ...
           (neg_num == 3)
                % Perform actions if all conditions are met
               %figure(1 + 5*(ind-1))
               figure(f1)
               % Subplot for he sparsity pattern of the SORTED adjacency matrix
               subplot(6, 6, (4+count1));
               count1 = count1 + 1 ;
               
                 
            % Subplot for he sparsity pattern of the SORTED adjacency matrix
           % subplot(6, 5, (4+file_num));
            spy(adjacency_matrix_sorted);

            % Create the dynamic title string using the extracted values
            dynamic_title = sprintf('P:%d,n:%d,a:%.2f,s:%d,l:%.3f,',...
                                    pos_num, neg_num, alpha_num, sm_num,lrate_num);

            % existing title
            existing_title = '';

            % Combine the existing title with the dynamic title
            combined_title = [existing_title, dynamic_title];

            title(combined_title);

            xlabel('V Ind');
            ylabel('V Ind');

            % Customize colors for the spy plot
            colormap(gca, [1, 1, 1; 0, 0, 0]); % White for zeros, Black for non-zeros

            % Keep the x and y tick labels as the sorted vertex indices
            xticks(1:size(adjacency_matrix_sorted, 1));
            yticks(1:size(adjacency_matrix_sorted, 1));
            xticklabels(num2str(sorted_vertex_indices));
            yticklabels(num2str(sorted_vertex_indices));


            xlabel('V Ind');
            ylabel('V Ind');

        end 
        
        %------------------------------------------------
        % 2
        % Check the conditions
        if (pos_num == 1) && ...
           (lrate_num == 0.025) && ...
           (alpha_num == 0 || alpha_num == 0.5 || alpha_num == 0.75) && ...
           (sm_num == 0 || sm_num == 1 || sm_num == 2 ) && ...
           (neg_num == 1 || neg_num == 2 || neg_num == 3)
            % Perform actions if all conditions are met
               %figure(2 + 5*(ind-1))
                              figure(f2)

               subplot(6, 5, (count2));
               count2 = count2 + 1 ;
               
                 
        % Subplot for he sparsity pattern of the SORTED adjacency matrix
           % subplot(6, 5, (4+file_num));
            spy(adjacency_matrix_sorted);

            % Create the dynamic title string using the extracted values
            dynamic_title = sprintf('P:%d,n:%d,a:%.2f,s:%d,l:%.3f,',...
                                    pos_num, neg_num, alpha_num, sm_num,lrate_num);

            % existing title
            existing_title = '';

            % Combine the existing title with the dynamic title
            combined_title = [existing_title, dynamic_title];

            title(combined_title);

            xlabel('V Ind');
            ylabel('V Ind');

            % Customize colors for the spy plot
            colormap(gca, [1, 1, 1; 0, 0, 0]); % White for zeros, Black for non-zeros

            % Keep the x and y tick labels as the sorted vertex indices
            xticks(1:size(adjacency_matrix_sorted, 1));
            yticks(1:size(adjacency_matrix_sorted, 1));
            xticklabels(num2str(sorted_vertex_indices));
            yticklabels(num2str(sorted_vertex_indices));


            xlabel('V Ind');
            ylabel('V Ind');
        end  
      
        %------------------------------------------------
        % 3
        % Check the conditions
        if (pos_num == 1 || pos_num == 2 || pos_num == 3) && ...
           (lrate_num == 0.025) && ...
           (alpha_num == 0 || alpha_num == 0.5) && ...
           (sm_num == 0 || sm_num == 1 || sm_num == 2 ) && ...
           (neg_num == 1 || neg_num == 2 || neg_num == 3)
            % Perform actions if all conditions are met
              % figure(3 + 5*(ind-1))
                              figure(f3)

               subplot(6, 6, (count3));
               count3 = count3 + 1 ;
               
                 
            % Subplot for he sparsity pattern of the SORTED adjacency matrix
           % subplot(6, 5, (4+file_num));
            spy(adjacency_matrix_sorted);

            % Create the dynamic title string using the extracted values
            dynamic_title = sprintf('P:%d,n:%d,a:%.2f,s:%d,l:%.3f,',...
                                    pos_num, neg_num, alpha_num, sm_num,lrate_num);

            % existing title
            existing_title = '';

            % Combine the existing title with the dynamic title
            combined_title = [existing_title, dynamic_title];

            title(combined_title);

            xlabel('V Ind');
            ylabel('V Ind');

            % Customize colors for the spy plot
            colormap(gca, [1, 1, 1; 0, 0, 0]); % White for zeros, Black for non-zeros

            % Keep the x and y tick labels as the sorted vertex indices
            xticks(1:size(adjacency_matrix_sorted, 1));
            yticks(1:size(adjacency_matrix_sorted, 1));
            xticklabels(num2str(sorted_vertex_indices));
            yticklabels(num2str(sorted_vertex_indices));


            xlabel('V Ind');
            ylabel('V Ind');
               
        end
        
        %------------------------------------------------
        % 4
        % Check the conditions
        if (pos_num == 1) && ...
           (lrate_num == 0.025 || lrate_num == 0.00625 || lrate_num == 0.1) && ...
           (alpha_num == 0 || alpha_num == 0.5) && ...
           (sm_num == 0 || sm_num == 1 || sm_num == 2) && ...
           (neg_num == 3)
            % Perform actions if all conditions are met
              % figure(4 + 5*(ind-1))
              figure(f4)
               subplot(6, 6, (count4));
               count4 = count4 + 1 ;
               
                 
            % Subplot for he sparsity pattern of the SORTED adjacency matrix
           % subplot(6, 5, (4+file_num));
            spy(adjacency_matrix_sorted);

            % Create the dynamic title string using the extracted values
            dynamic_title = sprintf('P:%d,n:%d,a:%.2f,s:%d,l:%.3f,',...
                                    pos_num, neg_num, alpha_num, sm_num,lrate_num);

            % existing title
            existing_title = '';

            % Combine the existing title with the dynamic title
            combined_title = [existing_title, dynamic_title];

            title(combined_title);

            xlabel('V Ind');
            ylabel('V Ind');

            % Customize colors for the spy plot
            colormap(gca, [1, 1, 1; 0, 0, 0]); % White for zeros, Black for non-zeros

            % Keep the x and y tick labels as the sorted vertex indices
            xticks(1:size(adjacency_matrix_sorted, 1));
            yticks(1:size(adjacency_matrix_sorted, 1));
            xticklabels(num2str(sorted_vertex_indices));
            yticklabels(num2str(sorted_vertex_indices));


            xlabel('V Ind');
            ylabel('V Ind');
        end
        
        %------------------------------------------------
        
        % 5
        % Check the conditions
        if (pos_num == 1 || pos_num == 2 || pos_num == 3) && ...
           (lrate_num == 0.025 || lrate_num == 0.00625 || lrate_num == 0.1) && ...
           (alpha_num == 0) && ...
           (sm_num == 0) && ...
           (neg_num == 1 || neg_num == 2 || neg_num == 3)
            % Perform actions if all conditions are met
               %figure(5 + 5*(ind-1))
               figure(f5)
               subplot(6, 5, (count5));
               count5 = count5 + 1 ;
               
                 
        % Subplot for he sparsity pattern of the SORTED adjacency matrix
       % subplot(6, 5, (4+file_num));
        spy(adjacency_matrix_sorted);

        % Create the dynamic title string using the extracted values
            dynamic_title = sprintf('P:%d,n:%d,a:%.2f,s:%d,l:%.3f,',...
                                pos_num, neg_num, alpha_num, sm_num,lrate_num);

        % existing title
        existing_title = '';

        % Combine the existing title with the dynamic title
        combined_title = [existing_title, dynamic_title];

        title(combined_title);

        xlabel('V Ind');
        ylabel('V Ind');

        % Customize colors for the spy plot
        colormap(gca, [1, 1, 1; 0, 0, 0]); % White for zeros, Black for non-zeros

        % Keep the x and y tick labels as the sorted vertex indices
        xticks(1:size(adjacency_matrix_sorted, 1));
        yticks(1:size(adjacency_matrix_sorted, 1));
        xticklabels(num2str(sorted_vertex_indices));
        yticklabels(num2str(sorted_vertex_indices));


        xlabel('V Ind');
        ylabel('V Ind');
        end
        
		%------------------------------------------------
        
        % 6
        % Check the conditions
        if (pos_num == 1 ) && ...
           (lrate_num == 0.025) && ...
           (alpha_num == 0) && ...
           (sm_num == 0 || sm_num == 1 || sm_num == 2 || sm_num == 3 || sm_num == 4 || sm_num == 5 || sm_num == 6 || sm_num == 7|| sm_num == 8|| sm_num == 9|| sm_num == 10) && ...
           (neg_num == 3)
            % Perform actions if all conditions are met
               %figure(5 + 5*(ind-1))
               figure(f6)
               subplot(4, 3, (count6));
               count6 = count6 + 1 ;
               
                 
        % Subplot for he sparsity pattern of the SORTED adjacency matrix
       % subplot(6, 5, (4+file_num));
        spy(adjacency_matrix_sorted);

        % Create the dynamic title string using the extracted values
            dynamic_title = sprintf('P:%d,n:%d,a:%.2f,s:%d,l:%.3f,',...
                                pos_num, neg_num, alpha_num, sm_num,lrate_num);

        % existing title
        existing_title = '';

        % Combine the existing title with the dynamic title
        combined_title = [existing_title, dynamic_title];

        title(combined_title);

        xlabel('V Ind');
        ylabel('V Ind');

        % Customize colors for the spy plot
        colormap(gca, [1, 1, 1; 0, 0, 0]); % White for zeros, Black for non-zeros

        % Keep the x and y tick labels as the sorted vertex indices
        xticks(1:size(adjacency_matrix_sorted, 1));
        yticks(1:size(adjacency_matrix_sorted, 1));
        xticklabels(num2str(sorted_vertex_indices));
        yticklabels(num2str(sorted_vertex_indices));


        xlabel('V Ind');
        ylabel('V Ind');
        end
        
        %------------------------------------------------
     toc
end
        
        figure(f1)
        %figure(1 + 5*(ind-1))
        % Save the current figure as an image
            % Set the figure to full-screen size
        set(gcf, 'Position', get(0, 'Screensize'));
        saveas(gcf, ['1_', input_file_prefix,  '.png']);
        
        
        figure(f2)

        %figure(2 + 5*(ind-1))
            % Set the figure to full-screen size
        set(gcf, 'Position', get(0, 'Screensize'));
        % Save the current figure as an image
        saveas(gcf, ['2_',input_file_prefix,  '.png']);
        
        figure(f3)

        %figure(3 + 5*(ind-1))
            % Set the figure to full-screen size
        set(gcf, 'Position', get(0, 'Screensize'));
        saveas(gcf, ['3_',input_file_prefix,  '.png']);

        
        figure(f4)

       % figure(4 + 5*(ind-1))
            % Set the figure to full-screen size
        set(gcf, 'Position', get(0, 'Screensize'));
        saveas(gcf, ['4_',input_file_prefix,  '.png']);

        figure(f5)

        %figure(5 + 5*(ind-1))
            % Set the figure to full-screen size
        set(gcf, 'Position', get(0, 'Screensize'));
        saveas(gcf, ['5_',input_file_prefix,  '.png']);
		
		
		figure(f6)

        %figure(5 + 5*(ind-1))
            % Set the figure to full-screen size
        set(gcf, 'Position', get(0, 'Screensize'));
        saveas(gcf, ['6_',input_file_prefix,  '.png']);
           
end

% Close the parallel pool
delete(pool);
