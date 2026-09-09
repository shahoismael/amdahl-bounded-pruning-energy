S = load('D:\claude_projects\ICIEAHS 2026\final_submission\results\info_mitbih.mat');
i = S.infoMit;
fprintf('splitMode: %s\n', i.splitMode);
fprintf('records: %s\n', strjoin(cellstr(i.recordIDs), ', '));
fprintf('TRAIN (%d): %s\n', numel(i.trainRecords), strjoin(cellstr(i.trainRecords), ', '));
fprintf('VAL   (%d): %s\n', numel(i.valRecords),   strjoin(cellstr(i.valRecords), ', '));
fprintf('TEST  (%d): %s\n', numel(i.testRecords),  strjoin(cellstr(i.testRecords), ', '));
fprintf('beats per record: %s\n', mat2str(i.beatsPerRecord));
fprintf('total beats: %d\n', i.numBeats);
disp(i.classCounts);

C = load('D:\claude_projects\ICIEAHS 2026\final_submission\results\info_chestxray.mat');
disp('--- chestxray info fields ---');
disp(fieldnames(C.infoChest));
disp(C.infoChest);
