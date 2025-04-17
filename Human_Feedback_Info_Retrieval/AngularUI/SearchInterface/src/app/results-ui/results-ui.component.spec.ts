import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ResultsUiComponent } from './results-ui.component';

describe('ResultsUiComponent', () => {
  let component: ResultsUiComponent;
  let fixture: ComponentFixture<ResultsUiComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ResultsUiComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ResultsUiComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
