classdef Satorbit
    methods(Static)
        function [t, coords] = read_orbits(path, processor)
            
            klass = {'char'}; %attr = {'nonempty'};
            attr = {};
            validateattributes(path,      klass, attr);
            validateattributes(processor, klass, attr);
            
            if strcmp(processor, 'doris')
                fid = Staux.sfopen(path, 'r');
                
                while true
                    line = fgetl(fid);
                    
                    if ~isempty(strfind(line, 'NUMBER_OF_DATAPOINTS:'))
                        break;
                    end
                end
                
                sline = strsplit(line, ':');
                ndata = str2num(sline{2});
                
                tcoords = fscanf(fid, '%f', [4, ndata])';
                
                fclose(fid);
                
                t = tcoords(:,1);
                coords = tcoords(:,2:end);
                
            elseif strcmp(processor, 'gamma')
                fid = Staux.sfopen(path, 'r');
                
                while true
                    line = fgetl(fid);
                    
                    if ~isempty(strfind(line, 'number_of_state_vectors:'))
                        break;
                    end
                end
                sline = strsplit(line, ':');
                ndata = str2num(sline{2});

                sline = strsplit(fgetl(fid), ':');
                sline = strsplit(sline{2});
                t_first = str2num(sline{1});

                sline = strsplit(fgetl(fid), ':');
                sline = strsplit(sline{2});
                t_step = str2num(sline{1});
                
                fclose(fid);
                
                
                t = t_first : t_step : t_first + ndata * t_step;
                coords = zeros(ndata, 3);
                
                for ii = 1:ndata
                   coords(ii,:) = ...
                   str2num(read_param(sprintf('state_vector_position_%d', ii), ...
                           path));
                end
                
                t = tcoords(:,1);
            else
                error('preprocessor should either be ''doris'' or ''gamma''!');
            end
        end % read_orbits
    
        function [] = fit_orbits(path, processor, varargin)
            
            klass = {'char'}; %attr = {'nonemtpy'};
            attr = {};
            validateattributes(path,      klass, attr);
            validateattributes(processor, klass, attr);
        
            args = struct('savepath', '', 'deg', 3, 'centered', false);
            args = Staux.parse_args(varargin, args, {'centered'});
            
            validateattributes(args.savepath, {'char'}, {});
            validateattributes(args.deg, {'numeric'}, ...
            {'real', 'scalar', 'integer', 'finite', 'nonnan'});
            
            [t, coords] = Satorbit.read_orbits(path, processor);
            
            ndata = numel(t);
            
            if args.centered
                t_mean = mean(t);
                t = t - mean_t;
                mean_coords = mean(coords, 1);
                coords = coords - repmat(mean_coords, ndata, 1);
            end
            
            % creating design matrix for coordinate fitting
            design = repmat(t, 1, args.deg - 1).^ repmat(args.deg:-1:2, ndata, 1);
            design = [design, t, ones(ndata, 1)];
            
            [fit, std, mse, s] = lscov(design, coords);
            
            if ~isempty(args.savepath)
                fid = sfopen(args.savepath, 'w');
                
                if args.centered
                    fprintf(fid, 'centered: 1\n');
                    fprintf(fid, 't_mean: %f\n', t_mean);
                    fprintf(fid, 'mean_coords: %f\n', mean_coords);
                else
                    fprintf(fid, 'centered: 0\n');
                end
                
                fprintf(fid, 'deg: %d\n', args.deg);
                fprintf(fid, 't_start: %f\n', t(1));
                fprintf(fid, 't_stop: %f\n', t(end));
                fprintf(fid, 'coeffs: %f\n', reshape(fit, 1, []));
                
                fclose(fid);
            end
        end
    end
end
