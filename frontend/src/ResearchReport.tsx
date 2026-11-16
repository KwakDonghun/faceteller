import {NOTICE, type ReportPassage, type ResearchReport} from './types';

// TEMPORARY research view (2026-11-16): the tables of experiments/feature17-20261109/run_person.py, as sent by the server.
// Nothing is computed or reworded here; the page only lays the report out.
const VALENCE: Record<string, string> = {좋음: 'favorable', 나쁨: 'unfavorable', 섞임: 'mixed', 중립: 'neutral'};
const two = (n: number | null | undefined) => typeof n === 'number' && Number.isFinite(n) ? n.toFixed(2) : '–';
const measured = (n: number | null) => n === null ? '–' : Math.abs(n) >= 10 ? n.toFixed(1) : n.toFixed(3);
/** "F17 턱 폭:좁음" -> "턱 폭: 좁음" (the feature code is internal). */
export const conditionText = (requirement: string[]) => requirement.map(c => c.split(' 또는 ').map(alt => alt.split('+').map(a => a.replace(/^F\d+[a-z]?\s+/, '').replace(':', ': ')).join(' 그리고 ')).join(' 또는 ')).join(' + ');
const Tone = ({value}: {value: string}) => <span className="valence" data-valence={VALENCE[value] || 'neutral'}>{value}</span>;

function PassageRows({rows, selected}: {rows: ReportPassage[]; selected?: Set<string>}) {
  return <div className="report-scroll"><table className="report-table report-passages"><thead><tr><th>조항 조건</th><th>신뢰도</th><th>풀이</th><th>문헌 번역</th>{selected && <th>5개 선정</th>}</tr></thead>
    <tbody>{rows.map(p => <tr key={p.representative}><td>{conditionText(p.requirement)}</td><td className="num">{two(p.G)}</td><td><Tone value={p.valence}/></td>
      <td>{p.translation || '번역 없음'}{p.source && <small className="report-source"> ({p.source})</small>}</td>{selected && <td>{selected.has(p.representative) ? '선정' : '5개가 먼저 참'}</td>}</tr>)}</tbody></table></div>;
}

export function ResearchReportView({report, notice, onReset}: {report: ResearchReport; notice?: string; onReset: () => void}) {
  const chosen = new Set(report.plan_C_selected.map(p => p.representative)), judgements = Object.entries(report.judgements);
  const confirmed = judgements.filter(([, j]) => j.status === '확정').length;
  const checks = Object.entries(report.cnn_check).flatMap(([feature, models]) => Object.entries(models).map(([model, c]) => ({feature, model, c})));
  return <section className="research-report" aria-labelledby="report-title">
    <div className="res-head"><h1 id="report-title">연구용 상세 결과</h1><span className="partchip">임시 화면</span></div>
    <p className="res-sub">판정 {judgements.length}개 중 확정 {confirmed}개 · 조건이 맞는 조항 {report.all_matching.length}개 · 선정 {report.plan_C_selected.length}개{report.structure ? ` · ${report.structure}` : ''}</p>

    <h2 className="section-title">고른 조항 (C안, 최대 5개)</h2>
    {report.plan_C_selected.length ? <PassageRows rows={report.plan_C_selected}/> : <p className="report-empty">조건이 맞는 조항이 없습니다.</p>}

    <h2 className="section-title">조건이 맞는 전체 조항 ({report.all_matching.length}개)</h2>
    {report.all_matching.length ? <PassageRows rows={report.all_matching} selected={chosen}/> : <p className="report-empty">조건이 맞는 조항이 없습니다.</p>}

    <h2 className="section-title">보류 때문에 나오지 못한 조항 ({report.blocked_by_hold.length}개)</h2>
    {report.blocked_by_hold.length ? <PassageRows rows={report.blocked_by_hold}/> : <p className="report-empty">보류 때문에 나오지 못한 조항은 없습니다.</p>}

    <h2 className="section-title">보류된 판정 ({report.held_judgements.length}개)</h2>
    {report.held_judgements.length ? <div className="report-scroll"><table className="report-table"><thead><tr><th>특징</th><th>보류된 판정</th><th>신뢰도</th><th>기준</th><th>조항이 안 나온 이유</th></tr></thead>
      <tbody>{report.held_judgements.map(h => <tr key={h.feature}><td>{h.name}</td><td>{h.class}</td><td className="num">{two(h.confidence)}</td><td className="num">{two(h.threshold)}</td><td>{h.reason}</td></tr>)}</tbody></table></div>
      : <p className="report-empty">보류된 판정이 없습니다.</p>}

    <h2 className="section-title">전체 판정 ({judgements.length}개)</h2>
    <div className="report-scroll"><table className="report-table"><thead><tr><th>특징</th><th>부위</th><th>방법</th><th>값</th><th>판정</th><th>신뢰도</th><th>기준</th><th>상태</th></tr></thead>
      <tbody>{judgements.map(([key, j]) => <tr key={key} data-status={j.status}><td>{j.name}</td><td>{j.region}</td><td>{j.method || '–'}</td><td className="num">{measured(j.value)}</td><td>{j.class || '–'}</td>
        <td className="num">{two(j.confidence)}</td><td className="num">{j.threshold === null ? '없음' : two(j.threshold)}</td><td><span className="report-status" data-status={j.status}>{j.status}</span></td></tr>)}</tbody></table></div>

    <h2 className="section-title">확인용 CNN (조항에는 쓰지 않음)</h2>
    {checks.length ? <div className="report-scroll"><table className="report-table"><thead><tr><th>특징</th><th>구조의 판정</th><th>확인용 모델</th><th>CNN 결과</th><th>CNN 신뢰도</th></tr></thead>
      <tbody>{checks.map(({feature, model, c}) => {const j = report.judgements[feature]; return <tr key={`${feature}-${model}`}><td>{j?.name || feature}</td><td>{j?.class ? `${j.class}${j.status === '확정' ? '' : ` (${j.status})`}` : '–'}</td><td>{model}</td><td>{c.class}</td><td className="num">{two(c.confidence)}</td></tr>;})}</tbody></table></div>
      : <p className="report-empty">확인용 CNN을 돌리지 않았습니다.</p>}

    {report.notes && report.notes.length > 0 && <ul className="report-notes">{report.notes.map(n => <li key={n}>{n}</li>)}</ul>}
    <p className="disclaimer">신뢰도는 판정이 경계에서 얼마나 떨어져 있는지를 나타내는 값이며 정답 확률이 아닙니다. {notice || NOTICE}</p>
    <div className="actions"><button className="button" onClick={onReset}>처음으로</button></div>
  </section>;
}
