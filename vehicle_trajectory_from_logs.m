% Данные из логов хранятся в мат-файлах ввиде массива.
% Временной интервал рисования
Time_range = [749393:1240200];

% Координаты центра масс
CenterFront = 3.7376;
CenterRear = 2.1024;

% Длина корпуса
L = CenterFront + CenterRear;

% Координата Х и Y
point_map(1,1) = 0;
point_map(1,2) = 0;

% Начальный угол направления
Angle_map(1) = 0;

% Временной шаг на каждой итерации
dt = gradient(seconds(All_Data_TT.Time));

% Счетчик остановок
station_counter = 0;
% Счетчик итераций
iter = 1;

for i = Time_range(1)+1 : Time_range(end)
    % Угол поворота передних колес
    a = All_Data_TT.Wheel_Angle(i);

    % Ограничение на минимальные углы
    if abs(a) < 1e-6
        a = sign(a) * 1e-6;
    end
    
    % Средняя скорость
    V = (All_Data_TT.Left_Speed(i) + All_Data_TT.Right_Speed(i)) / 2;
    if V == 0
        continue
    end

    iter = iter + 1;

    % Обновляем угол направления с использованием кинематической модели
    Angle_map(iter) = Angle_map(iter-1) + (V / L) * tan(a) * dt(i);

    % Обновляем координаты
    point_map(iter,1) = point_map(iter-1,1) + V * cos(Angle_map(iter)) * dt(i);
    point_map(iter,2) = point_map(iter-1,2) + V * sin(Angle_map(iter)) * dt(i);

    % Проверка на остановки
    if All_Data_TT.DoorsOpen_CE(i) + All_Data_TT.DoorsOpen_CD(i) + All_Data_TT.DoorsOpen_CC(i) > 0
        station_counter = station_counter + 1;
        station_point(station_counter,1) = point_map(iter,1);
        station_point(station_counter,2) = point_map(iter,2);
    end
    
    % Визуализация каждые 100 итераций
    if mod(iter,100) == 0
        % plot(point_map(:,1)/1000, point_map(:,2)/1000)
    end
end
