import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SearchUIComponent } from './search-ui.component';

describe('SearchUIComponent', () => {
  let component: SearchUIComponent;
  let fixture: ComponentFixture<SearchUIComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ SearchUIComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(SearchUIComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
