import {beforeAll, describe, expect, it} from 'vitest';
import {renderToStaticMarkup} from 'react-dom/server';
import {Results} from '../src/Results';
import {conditionText} from '../src/ResearchReport';
import {validateResult, validReport} from '../src/api';
import type {AnalysisResult, ResearchReport} from '../src/types';

// Authored synthetic values only; no real person's result. Temporary research view (2026-11-16).
const passage = (id: string, requirement: string[], valence: string, G: number, feature: string, cls: string) =>
  ({representative: id, requirement, valence, G, regions: ['합성 부위'], features: {[feature]: {class: cls, confidence: G, held: false}}, translation: `합성 번역 ${id}`, source: '합성 문헌'});
const report = (): ResearchReport => ({schema: 'faceteller-research-report-v1', structure: '합성 구조',
  judgements: {F09: {name: '코 길이', region: '코', method: '정면 수치', value: .3712, class: '보통', confidence: .81, threshold: .755, status: '확정'},
    F15: {name: '인중 길이', region: '인중', method: 'CNN', value: null, class: '보통', confidence: .55, threshold: .6, status: '보류'},
    F20: {name: '광대 돌출', region: '광대', method: 'VGGT 맞춤식', value: -.5, class: '보통', confidence: .7, threshold: null, status: '확정'}},
  plan_C_selected: [passage('S-1', ['F05 눈썹 휨:일자'], '좋음', .91, 'F05', '일자')],
  all_matching: [passage('S-1', ['F05 눈썹 휨:일자'], '좋음', .91, 'F05', '일자'), passage('S-2', ['F18 눈썹 숱:성김', 'F04 눈썹 꼬리 길이:긴 편+F06 눈썹-눈 간격:넓음'], '나쁨', .77, 'F18', '성김')],
  blocked_by_hold: [], held_judgements: [{feature: 'F15', name: '인중 길이', class: '보통', confidence: .55, threshold: .6, reason: '이 칸을 조건으로 쓰는 조항이 없음'}],
  cnn_check: {F09: {'합성 확인 모델': {class: '김', confidence: .66}}}});
const result = (r: unknown): AnalysisResult => validateResult({status: 'COMPLETE', items: [], passages: [], research_report: r});
const render = (r: AnalysisResult) => renderToStaticMarkup(<Results result={r} onPreview={() => {}} onReset={() => {}}/>);

beforeAll(() => {(globalThis as unknown as {window: object}).window = {};});

describe('temporary research report view', () => {
  it('replaces the service cards and shows every table of the report', () => {
    const html = render(result(report()));
    expect(html).toContain('연구용 상세 결과');
    expect(html).not.toContain('생김새로 본 관상 풀이');
    for (const text of ['고른 조항 (C안, 최대 5개)', '조건이 맞는 전체 조항 (2개)', '보류 때문에 나오지 못한 조항은 없습니다.', '보류된 판정 (1개)', '전체 판정 (3개)', '확인용 CNN (조항에는 쓰지 않음)']) expect(html).toContain(text);
    expect(html).toContain('눈썹 휨: 일자'); expect(html).toContain('합성 번역 S-1'); expect(html).toContain('(합성 문헌)');
    expect(html).toContain('5개가 먼저 참'); expect(html).toContain('이 칸을 조건으로 쓰는 조항이 없음');
    expect(html).toContain('<td class="num">0.55</td>'); expect(html).toContain('<td class="num">없음</td>'); expect(html).toContain('합성 확인 모델');
    expect(html).toContain('정답 확률이 아닙니다');
  });

  it('writes conditions without the internal feature codes', () => {
    expect(conditionText(['F17 턱 폭:좁음'])).toBe('턱 폭: 좁음');
    expect(conditionText(['F18 눈썹 숱:짙음', 'F04 눈썹 꼬리 길이:긴 편+F06 눈썹-눈 간격:넓음'])).toBe('눈썹 숱: 짙음 + 눈썹 꼬리 길이: 긴 편 그리고 눈썹-눈 간격: 넓음');
  });

  it('drops a malformed report instead of repairing it, and keeps the normal view', () => {
    const bad = {...report(), plan_C_selected: [{representative: 'x'}]};
    expect(validReport(bad)).toBeNull();
    expect(validReport({...report(), schema: 'other'})).toBeNull();
    expect(validReport({...report(), plan_C_selected: Array.from({length: 6}, (_, i) => passage(`S-${i}`, ['F05 눈썹 휨:일자'], '좋음', .91, 'F05', '일자'))})).toBeNull();
    const html = render(result(bad));
    expect(html).not.toContain('연구용 상세 결과'); expect(html).toContain('생김새로 본 관상 풀이');
  });

  it('results without a report are untouched', () => {
    expect(validateResult({status: 'COMPLETE', items: [], passages: []}).research_report).toBeUndefined();
  });
});
