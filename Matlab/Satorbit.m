classdef Satorbit
    methods(Static)
        function [t, coords] = read_orbits(path, processor)
            
            klass = {'char'}; attr = {'nonempty'};
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
                    
                    if Staux.strin('number_of_state_vectors:', line)
                        break;
                    end
                    
                    if line == -1
                        error(['End of file reached without ', ...
                               '`number_of_state_vectors` parameter']);
                    end
                end
                sline = strsplit(line, ':');
                ndata = str2num(sline{2});

                sline = strsplit(fgetl(fid), ':');
                sline = strsplit(sline{2}, 's');
                t_first = str2num(sline{1});

                sline = strsplit(fgetl(fid), ':');
                sline = strsplit(sline{2}, 's');
                t_step = str2num(sline{1});
                
                fclose(fid);
                
                t = t_first : t_step : t_first + (ndata - 1) * t_step;
                coords = zeros(ndata, 3);
                
                for ii = 1:ndata
                    pos = Staux.read_param(sprintf('state_vector_position_%d', ii), ...
                           path);
                    spos = strsplit(pos, 'm');
                    coords(ii,:) = str2num(spos{1});
                end
            else
                error('preprocessor should either be ''doris'' or ''gamma''!');
            end
        end % read_orbits
    
        function [] = fit_orbits(path, processor, varargin)
            
            klass = {'char'}; attr = {'nonemtpy'};
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
            
            if isrow(t)
                t = t';
            end
            
            if args.centered
                t_mean = mean(t);
                t = t - t_mean;
                mean_coords = mean(coords, 1);
                coords = coords - repmat(mean_coords, ndata, 1);
            end
            
            % creating design matrix for coordinate fitting
            design = repmat(t, 1, args.deg - 1).^ repmat(args.deg:-1:2, ndata, 1);
            design = [design, t, ones(ndata, 1)];
            
            [fit, std, mse] = lscov(design, coords);
            
            if ~isempty(args.savepath)
                fid = sfopen(args.savepath, 'w');
                
                if args.centered
                    fprintf(fid, 'centered: 1\n');
                    fprintf(fid, 't_mean: %f\n', t_mean);
                    fprintf(fid, 'mean_coords: ');
                    fprintf(fid, '%f ', mean_coords);
                    fprintf(fid, '\n');
                else
                    fprintf(fid, 'centered: 0\n');
                end
                
                fprintf(fid, 'deg: %d\n', args.deg);
                fprintf(fid, 't_start: %f\n', t(1));
                fprintf(fid, 't_stop: %f\n', t(end));
                fprintf(fid, 'coeffs: ');
                fprintf(fid, '%f ', reshape(fit, 1, []));
                fprintf(fid, '\n');
                
                fclose(fid);
            end
        end % fit_orbits
        
        function read_fit(path)
            klass = {'char'}; attr = {'nonemtpy'};
            attr = {};
            validateattributes(path, klass, attr);
            
            fid = Staux.sfopen(path, 'r');
            
            centered = fscanf(fid, 'centered: %d\n');
            
            if centered
                t_mean = fscanf(fid, 't_mean: %f\n');
                mean_coords = fscanf(fid, 'mean_coords: %f %f %f\n');
            end
            
            deg = fscanf(fid, 'deg: %d\n')
            t_start = fscanf(fid, 't_start: %f\n')
            t_stop  = fscanf(fid, 't_stop: %f\n')
            
            fclose(fid);
        end % read_fit
    end
end
